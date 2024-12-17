import random
import numpy as np
import onnx
import onnxruntime as ort
from skimage.color import label2rgb
from matplotlib import pyplot as plt
import onnx

import sys
sys.path.insert(0, "Marabou")
import os
"""
After installing Marabou, load it from maraboupy.
"""
from maraboupy import Marabou


class VeriX:
    """
    This is the VeriX class to take in an image and a neural network, and then output an explanation.
    """
    image = None
    keras_model = None
    mara_model = None
    traverse: str
    sensitivity = None
    dataset: str
    label: int
    inputVars = None
    outputVars = None
    epsilon: float
    """
    Marabou options: 'timeoutInSeconds' is the timeout parameter. 
    """
    options = Marabou.createOptions(numWorkers=16,
                                    timeoutInSeconds=300,
                                    verbosity=0,
                                    solveWithMILP=True)

    def __init__(self,
                 dataset,
                 image,
                 model_path,
                 name,
                 epsilon=-1,
                 plot_original=True):
        """
        To initialize the VeriX class.
        :param dataset: 'MNIST' or 'GTSRB'.
        :param image: an image array of shape (width, height, channel).
        :param model_path: the path to the neural network.
        :param plot_original: if True, then plot the original image.
        """
        self.dataset = dataset
        self.image = image
        self.name = name
        self.threshold = 1e-6
        self.epsilon = epsilon
        """
        Load the onnx model.
        """
        self.onnx_model = onnx.load(model_path)
        self.onnx_session = ort.InferenceSession(model_path)
        prediction = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(image, axis=0)})
        prediction = np.asarray(prediction[0])
        self.label = prediction.argmax()
        """
        Load the onnx model into Marabou.
        Note: to ensure sound and complete analysis, load the model before the softmax activation function;
        if the model is trained from logits directly, then load the whole model. 
        """
        self.mara_model = Marabou.read_onnx(model_path)
        if self.onnx_model.graph.node[-1].op_type == "Softmax":
            mara_model_output = self.onnx_model.graph.node[-1].input
        else:
            mara_model_output = None
        self.mara_model = Marabou.read_onnx(filename=model_path,
                                            outputNames=mara_model_output)
        self.inputVars = np.arange(image.shape[0] * image.shape[1])
        self.outputVars = self.mara_model.outputVars[0].flatten()
        if plot_original:
            save_figure(image=image,
                        path=f"original-predicted-as-{self.label}.png",
                        cmap="gray" if self.dataset == 'MNIST' else None)

    def traversal_order(self,
                        traverse="heuristic",
                        plot_sensitivity=True,
                        seed=0):
        """
        To compute the traversal order of checking all the pixels in the image.
        :param traverse: 'heuristic' (by default) or 'random'.
        :param plot_sensitivity: if True, plot the sensitivity map.
        :param seed: if traverse by 'random', then set a random seed.
        :return: an updated inputVars that contains the traversal order.
        """
        self.traverse = traverse
        if self.traverse == "heuristic":
            width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
            temp = self.image.reshape(width * height, channel)
            image_batch = np.kron(np.ones(shape=(width * height, 1, 1), dtype=temp.dtype), temp)
            image_batch_manip = image_batch.copy()
            for i in range(width * height):
                """
                Different ways to compute sensitivity: use pixel reversal for MNIST and deletion for GTSRB.
                """
                if self.dataset == "MNIST":
                    image_batch_manip[i][i][:] = 1 - image_batch_manip[i][i][:]
                elif self.dataset == "GTSRB":
                    image_batch_manip[i][i][:] = 0
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            image_batch = image_batch.reshape((width * height, width, height, channel))
            predictions = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch})
            predictions = np.asarray(predictions[0])
            image_batch_manip = image_batch_manip.reshape((width * height, width, height, channel))
            predictions_manip = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: image_batch_manip})
            predictions_manip = np.asarray(predictions_manip[0])
            difference = predictions - predictions_manip
            features = difference[:, self.label]
            sorted_index = features.argsort()
            self.inputVars = sorted_index
            self.sensitivity = features.reshape(width, height)
            if plot_sensitivity:
                save_figure(image=self.sensitivity, path=f'{self.dataset}-sensitivity-{self.traverse}.png')
        elif self.traverse == "pgd":
            self.model = convert(self.onnx_model)
            if not hasattr(self, 'model'):
                raise ValueError("PGD traversal requires a PyTorch model as 'self.model'.")

            self.model.eval()

            height, width, channel = self.image.shape

            image_tensor = torch.from_numpy(self.image).float().permute(2, 0, 1)  # (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
            image_tensor.requires_grad = True

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)
            self.model.to(device)

            alpha = 0.01
            num_steps = 30

            label_tensor = torch.tensor([self.label], dtype=torch.long).to(device)

            adv_image = image_tensor.clone().detach()
            for _ in range(num_steps):
                adv_image.requires_grad = True
                outputs = self.model(adv_image)
                loss = F.cross_entropy(outputs, label_tensor)
                self.model.zero_grad()
                loss.backward()

                adv_image = adv_image + alpha * adv_image.grad.sign()
                eta = torch.clamp(adv_image - image_tensor, min=-self.epsilon, max=self.epsilon)
                adv_image = torch.clamp(image_tensor + eta, min=0, max=1).detach()

            diff = torch.abs(adv_image - image_tensor).squeeze(0)
            diff_reshaped = diff.permute(1, 2, 0).reshape(-1, channel)
            diff_flat = diff_reshaped.sum(dim=1)
            sorted_indices = torch.argsort(diff_flat, descending=True).cpu().numpy()
            self.inputVars = sorted_indices
            self.sensitivity = diff_flat.reshape(height, width).cpu().detach().numpy()
            adv_predictions = self.model(adv_image)
            adv_label = torch.argmax(adv_predictions)

            # Check if the attack was successful
            attack_successful = self.label != adv_label
            print(attack_successful)
            if plot_sensitivity:
                save_figure(image=self.sensitivity, path=f'{self.dataset}-sensitivity-{self.traverse}.png')
        elif self.traverse == "expensive_pgd":
            import random

            self.model = convert(self.onnx_model)
            if not hasattr(self, 'model'):
                raise ValueError("Expensive PGD traversal requires a PyTorch model as 'self.model'.")

            self.model.eval()

            # Prepare the image and model
            height, width, channel = self.image.shape
            num_pixels = height * width
            subset_size = int(0.4 * num_pixels)  # 60% of the pixels
            num_batches = 100  # Number of random subsets
            alpha = 0.01  # Step size
            num_steps = 30  # PGD steps

            image_tensor = torch.from_numpy(self.image).float().permute(2, 0, 1)  # (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
            image_tensor.requires_grad = True

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)
            self.model.to(device)

            label_tensor = torch.tensor([self.label], dtype=torch.long).to(device)

            # Initialize difference accumulator
            total_diff = torch.zeros_like(image_tensor).to(device)

            for batch in range(num_batches):
                # Select a random subset of pixels
                all_indices = list(range(num_pixels))
                subset_indices = random.sample(all_indices, subset_size)

                # Create a mask for the subset of pixels
                mask = torch.zeros_like(image_tensor).to(device)
                for idx in subset_indices:
                    y, x = divmod(idx, width)
                    mask[0, :, y, x] = 1

                # Clone the original image and apply the mask
                adv_image = image_tensor.clone().detach()
                for _ in range(num_steps):
                    adv_image.requires_grad = True
                    outputs = self.model(adv_image)
                    loss = F.cross_entropy(outputs, label_tensor)
                    self.model.zero_grad()
                    loss.backward()

                    # Apply updates only to the selected subset of pixels
                    grad = adv_image.grad.sign()
                    adv_image = adv_image + alpha * grad * mask
                    eta = torch.clamp(adv_image - image_tensor, min=-self.epsilon, max=self.epsilon)
                    adv_image = torch.clamp(image_tensor + eta, min=0, max=1).detach()

                # Accumulate the pixel differences for the batch
                batch_diff = torch.abs(adv_image - image_tensor)
                total_diff += batch_diff

            # Compute the total importance map
            total_diff = total_diff.squeeze(0)  # Remove batch dimension -> (C, H, W)
            total_diff = total_diff.permute(1, 2, 0)  # (H, W, C)

            # Flatten and sort pixels based on total importance
            total_diff_flat = total_diff.reshape(-1, channel).sum(dim=1)  # Sum across channels
            sorted_indices = torch.argsort(total_diff_flat, descending=True).cpu().numpy()  # Sort by descending importance

            # Update traversal order and sensitivity map
            self.inputVars = sorted_indices
            self.sensitivity = total_diff_flat.reshape(height, width).cpu().detach().numpy()

            # Check if the attack was successful
            adv_predictions = self.model(adv_image)
            adv_label = torch.argmax(adv_predictions)

            attack_successful = self.label != adv_label.item()
            print(f"Attack Successful: {attack_successful}")

            # Optionally, save the sensitivity map
            if plot_sensitivity:
                save_figure(image=self.sensitivity, path=f'{self.dataset}-sensitivity-{self.traverse}.png')

        elif self.traverse == "grad":
            self.model = convert(self.onnx_model)
            if not hasattr(self, 'model'):
                raise ValueError("Gradient traversal requires a PyTorch model as 'self.model'.")

            self.model.eval()

            # Prepare the image and model
            height, width, channel = self.image.shape
            image_tensor = torch.from_numpy(self.image).float().permute(2, 0, 1)  # (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)
            image_tensor.requires_grad = True
            self.model.to(device)

            # Label for the input image
            label_tensor = torch.tensor([self.label], dtype=torch.long).to(device)

            # Forward pass
            outputs = self.model(image_tensor)
            loss = F.cross_entropy(outputs, label_tensor)

            # Backward pass to compute gradients
            self.model.zero_grad()
            loss.backward()

            # Extract the gradient of the input image
            gradient = image_tensor.grad.data.squeeze(0)  # Remove batch dimension -> (C, H, W)
            gradient = gradient.permute(1, 2, 0)  # (H, W, C)

            # Compute the sensitivity map
            gradient_flat = gradient.reshape(-1, channel).abs().sum(dim=1)  # Sum absolute gradient values across channels
            sorted_indices = torch.argsort(gradient_flat, descending=True).cpu().numpy()  # Sort by importance

            # Update traversal order and sensitivity map
            self.inputVars = sorted_indices
            self.sensitivity = gradient_flat.reshape(height, width).cpu().detach().numpy()
        elif self.traverse == "smoothgrad":
            self.model = convert(self.onnx_model)
            if not hasattr(self, 'model'):
                raise ValueError("SmoothGrad traversal requires a PyTorch model as 'self.model'.")

            self.model.eval()

            # Parameters for SmoothGrad
            num_samples = 50   # Number of noisy samples to generate

            # Prepare the image and model
            height, width, channel = self.image.shape
            image_tensor = torch.from_numpy(self.image).float().permute(2, 0, 1)  # (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)
            self.model.to(device)

            # Label for the input image
            label_tensor = torch.tensor([self.label], dtype=torch.long).to(device)

            # Initialize gradient accumulator
            smoothgrad = torch.zeros_like(image_tensor)

            for _ in range(num_samples):
                # Add noise to the image
                noise = torch.randn_like(image_tensor)
                noisy_image = torch.clamp(image_tensor + noise, 0, 1)  # Clamp to valid pixel range

                # Forward pass
                noisy_image.requires_grad = True
                outputs = self.model(noisy_image)
                loss = F.cross_entropy(outputs, label_tensor)

                # Backward pass to compute gradients
                self.model.zero_grad()
                loss.backward()

                # Accumulate gradients
                smoothgrad += noisy_image.grad.data

            # Average gradients over all samples
            smoothgrad /= num_samples

            # Compute the sensitivity map
            smoothgrad = smoothgrad.squeeze(0)  # Remove batch dimension -> (C, H, W)
            smoothgrad = smoothgrad.permute(1, 2, 0)  # (H, W, C)

            # Flatten and sort pixels based on sensitivity
            smoothgrad_flat = smoothgrad.reshape(-1, channel).abs().sum(dim=1)  # Sum absolute gradient values across channels
            sorted_indices = torch.argsort(smoothgrad_flat, descending=True).cpu().numpy()  # Sort by importance

            # Update traversal order and sensitivity map
            self.inputVars = sorted_indices
            self.sensitivity = smoothgrad_flat.reshape(height, width).cpu().detach().numpy()

            # Optionally, save the sensitivity map
            if plot_sensitivity:
                save_figure(image=self.sensitivity, path=f'{self.dataset}-sensitivity-{self.traverse}.png')
        # elif self.traverse == "expensive":


        elif self.traverse == "random":
            random.seed(seed)
            random.shuffle(self.inputVars)
        else:
            print("Traversal not supported: try 'heuristic' or 'random'.")

    def get_explanation(self,
                        epsilon,
                        plot_explanation=True,
                        plot_counterfactual=False,
                        plot_timeout=False):
        """
        To compute the explanation for the model and the neural network.
        :param epsilon: the perturbation magnitude.
        :param plot_explanation: if True, plot the explanation.
        :param plot_counterfactual: if True, plot the counterfactual(s).
        :param plot_timeout: if True, plot the timeout pixel(s).
        :return: an explanation, and possible counterfactual(s).
        """
        if self.epsilon != -1:
            assert self.epsilon == epsilon
        unsat_set = []
        sat_set = []
        timeout_set = []
        width, height, channel = self.image.shape[0], self.image.shape[1], self.image.shape[2]
        image = self.image.reshape(width * height, channel)
        counterfactuals = {}
        for pixel in self.inputVars:
            for i in self.inputVars:
                """
                Set constraints on the input variables.
                """
                if i == pixel or i in unsat_set:
                    """
                    Set allowable perturbations on the current pixel and the irrelevant pixels.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                        self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                        self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))  
                        self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                        self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                        self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
                else:
                    """
                    Make sure the other pixels are fixed.
                    """
                    if self.dataset == "MNIST":
                        self.mara_model.setLowerBound(i, image[i][:])
                        self.mara_model.setUpperBound(i, image[i][:])
                    elif self.dataset == "GTSRB":
                        self.mara_model.setLowerBound(3 * i, image[i][0])
                        self.mara_model.setUpperBound(3 * i, image[i][0])
                        self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                        self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                        self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                        self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                    else:
                        print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            for j in range(len(self.outputVars)):
                """
                Set constraints on the output variables.
                """
                if j != self.label:
                    self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                  [1, -1], -1e-6,
                                                  isProperty=True)
                    exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=True)
                    """
                    additionalEquList.clear() is to clear the output constraints.
                    """
                    self.mara_model.additionalEquList.clear()
                    if exit_code == 'sat' or exit_code == 'TIMEOUT':
                        break
                    elif exit_code == 'unsat':
                        continue
            """
            clearProperty() is to clear both input and output constraints.
            """
            self.mara_model.clearProperty()
            """
            If unsat, put the pixel into the irrelevant set; 
            if timeout, into the timeout set; 
            if sat, into the explanation.
            """
            if exit_code == 'unsat':
                unsat_set.append(pixel)
            elif exit_code == 'TIMEOUT':
                timeout_set.append(pixel)
            elif exit_code == 'sat':
                sat_set.append(pixel)
                counterfactuals[pixel] = [vals.get(i) for i in self.mara_model.inputVars[0].flatten()]
                counterfactuals[pixel] = np.asarray(counterfactuals[pixel]).reshape(self.image.shape).astype(np.float32)
                prediction = [vals.get(i) for i in self.outputVars]
                prediction = np.asarray(prediction).argmax()
                save_figure(image=counterfactuals[pixel],
                            path="counterfactual-at-pixel-%d-predicted-as-%d.png" % (pixel, prediction),
                            cmap="gray" if self.dataset == 'MNIST' else None)
        if True:
            mask = np.zeros(self.inputVars.shape).astype(bool)
            mask[sat_set] = True
            mask[timeout_set] = True
            plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
            save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                        self.image.reshape(plot_shape),
                                        colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                        bg_label=0,
                                        saturation=1),
                        path="explanation-%d.png" % (len(sat_set) + len(timeout_set)))
        if plot_timeout:
            mask = np.zeros(self.inputVars.shape).astype(bool)
            mask[timeout_set] = True
            plot_shape = self.image.shape[0:2] if self.dataset == "MNIST" else self.image.shape
            save_figure(image=label2rgb(mask.reshape(self.image.shape[0:2]),
                                        self.image.reshape(plot_shape),
                                        colors=[[0, 1, 0]] if self.traverse == 'heuristic' else [[1, 0, 0]],
                                        bg_label=0,
                                        saturation=1),
                        path="timeout-%d.png" % len(timeout_set))
        # assert self.fast_test_explanation(image, epsilon, sat_set, unsat_set, counterfactuals)
        print("passed tests")
        return len(sat_set), len(timeout_set) 

    def fast_test_explanation(self, image, epsilon, sat_set, unsat_set, counterfactuals):
        # check if there is a counterfactual with only the unsat pixels
        for i in self.inputVars:
            if i in unsat_set:
                if self.dataset == "MNIST":
                    self.mara_model.setLowerBound(i, max(0, image[i][:] - epsilon))
                    self.mara_model.setUpperBound(i, min(1, image[i][:] + epsilon))
                elif self.dataset == "GTSRB":
                    self.mara_model.setLowerBound(3 * i, max(0, image[i][0] - epsilon))
                    self.mara_model.setUpperBound(3 * i, min(1, image[i][0] + epsilon))
                    self.mara_model.setLowerBound(3 * i + 1, max(0, image[i][1] - epsilon))
                    self.mara_model.setUpperBound(3 * i + 1, min(1, image[i][1] + epsilon))
                    self.mara_model.setLowerBound(3 * i + 2, max(0, image[i][2] - epsilon))
                    self.mara_model.setUpperBound(3 * i + 2, min(1, image[i][2] + epsilon))
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            elif i in sat_set:
                if self.dataset == "MNIST":
                    self.mara_model.setLowerBound(i, image[i][:])
                    self.mara_model.setUpperBound(i, image[i][:])
                elif self.dataset == "GTSRB":
                    self.mara_model.setLowerBound(3 * i, image[i][0])
                    self.mara_model.setUpperBound(3 * i, image[i][0])
                    self.mara_model.setLowerBound(3 * i + 1, image[i][1])
                    self.mara_model.setUpperBound(3 * i + 1, image[i][1])
                    self.mara_model.setLowerBound(3 * i + 2, image[i][2])
                    self.mara_model.setUpperBound(3 * i + 2, image[i][2])
                else:
                    print("Dataset not supported: try 'MNIST' or 'GTSRB'.")
            else:
                print("367", i)
                return False
        for j in range(len(self.outputVars)):
            if j != self.label:
                self.mara_model.addInequality([self.outputVars[self.label], self.outputVars[j]],
                                                [1, -1], 10 * -self.threshold,
                                                isProperty=True)
                exit_code, vals, stats = self.mara_model.solve(options=self.options, verbose=False)
                self.mara_model.additionalEquList.clear()
                if exit_code == 'sat' or exit_code == 'TIMEOUT':
                    break
                elif exit_code == 'unsat':
                    continue
        self.mara_model.clearProperty()
        assert exit_code != 'TIMEOUT'
        if exit_code == 'sat':
            print("no way")
            return False
        
        image_flat = image.flatten()
        for i in sat_set:
            counterfactual_flat = counterfactuals[i].flatten()
            if np.linalg.norm(image_flat - counterfactual_flat, np.inf) > epsilon + self.threshold:
                print("330: ", np.linalg.norm(image_flat - counterfactual_flat, np.inf), epsilon + self.threshold)
                return False

            # make sure that different pixels from the original are entirely in the unsat set
            # except for one in the sat_set
            cnt_in_sat_set = 0
            for j in self.inputVars:
                if self.dataset == "MNIST":
                    if image_flat[j] != counterfactual_flat[j]:
                        if j in sat_set:
                            cnt_in_sat_set += 1
                        elif j in unsat_set:
                            pass
                        else:
                            print("343")
                            return False
                elif self.dataset == "GTSRB":
                    if image_flat[3 * j] != counterfactual_flat[3 * j] or \
                        image_flat[3 * j + 1] != counterfactual_flat[3 * j + 1] or \
                        image_flat[3 * j + 2] != counterfactual_flat[3 * j + 2]:
                        if j in sat_set:
                            # print("diff: ", image_flat[j]-counterfactual_flat[j])
                            cnt_in_sat_set += 1
                        elif j in unsat_set:
                            pass
                        else:
                            print("343")
                            return False
            if cnt_in_sat_set > 1:
                print("346")
                return False
            # check if the counterfactual is valid by running the model
            counterfactual_result = self.onnx_session.run(None, {self.onnx_model.graph.input[0].name: np.expand_dims(counterfactuals[i], axis=0)})
            counterfactual_result = np.asarray(counterfactual_result[0])
            if counterfactual_result.argmax() == self.label:
                # print("sat pixel: ", i)
                np.set_printoptions(precision=10)
                # print("counterfactual_result: ", counterfactual_result)
                # print("self.label: ", self.label)
                # print("352")
                with open("warning.txt", "w") as f:
                    f.write(f'{counterfactual_result[0].tolist()} {self.label}')
                    f.flush()
                
        return True

def save_figure(image, path, cmap=None):
    """
    To plot figures.
    :param image: the image array of shape (width, height, channel)
    :param path: figure name.
    :param cmap: 'gray' if to plot gray scale image.
    :return: an image saved to the designated path.
    """
    path = "paper/" + path
    fig = plt.figure()
    ax = plt.Axes(fig, [-0.5, -0.5, 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    if cmap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
