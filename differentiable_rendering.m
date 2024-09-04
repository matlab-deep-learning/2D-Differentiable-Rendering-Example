%% 2D Differentiable Rendering in MATLAB 

% This example shows how to perform differentiable vector graphics rendering 
% in MATLAB®. It makes use of dlarray for auto-differentiation, and employs 
% signed distance functions and sigmoid smoothing to ensure the rasterization 
% pipeline is continuous and differentiable. First, we look at how to perform 
% differentiable rendering. Then, we use the differentiable rendering pipeline 
% to optimize shape parameters to fit a target image.

% Copyright 2024 The MathWorks, Inc

%% Optimizing rasterized shapes to approximate an image
% Choose one of the following examples to see how we can optimize shape parameters 
% to render an image as similar as possible to a target image.

% In circles example, we optimize the position, size and color of two circles 
% to perfectly match a target image.
% 
% In gaussians example, we optimize the mean, sigma, and color of a large number 
% of Gaussian functions to approximate a target image.
% 
% In rectangles example, we optimize the position, size and color of many rectangles 
% to approximate a target image. 

example = "circles";
% example = "gaussians";
% example = "rectangles";

%% Load a target image to approximate

% Here we load the image we want to approximate with our shapes. We also pre-compute 
% a grid of normalized pixel coordinates to be used in the rendering function. 

% Load target image
if strcmp(example, "rectangles") || strcmp(example, "gaussians")
    target_image = im2double(imread('bernie.jpg'));
elseif strcmp(example, "circles")
    target_image = im2double(imread('circle.png'));
end

% Convert to dlarray
target_image = dlarray(target_image, 'SSCB'); 

% Precompute a pixel grid
[image_height, image_width, ~, ~] = size(target_image);
[X, Y] = ndgrid(linspace(0,1,image_height), linspace(0,1,image_width));
X = dlarray(X, 'SSCB');
Y = dlarray(Y, 'SSCB');

%% Define the main hyperparameters

% The hyperparameters are the type of shape to use (rectangle, circle or Gaussian), 
% and the number of them to use. 

switch example
    case "rectangles"
        % Select the type of shape and number of them to use
        shape_type = "rectangle";
        num_shapes = 75; 
    case "circles"
        shape_type = "circle"; 
        num_shapes = 2; 
    case "gaussians"
        shape_type = "gaussians";
        num_shapes = 150;
end

%% Initialize the parameters

% Start by choosing a random initialization of the shape parameters. _Note: 
% To find a particularly good set of initial values one could repeat the random 
% initialization process multiple times and choose the set that result in the 
% lowest loss function. This is not implemented here_.
% 
% *The shapes are each parameterized by three variables:*
% 
% For *rectangles*, _center_ is the coordinate of the center of the rectangle, 
% _dimensions_ is the width and height of the rectangle, and _color_ is the RGB 
% color.
% 
% For circle*s*, _center_ is the coordinate of the center of the circle, _dimensions_ 
% is the width and height of the circle, and _color_ is the RGB color.
% 
% For *Gaussians*, _center_ is the coordinate of the center (i.e. the mean), 
% _dimensions_ is the standard deviation in the x and y directions, and _color_ 
% is the RGB color 

switch example
    case {"rectangles", "gaussians"}
    % Make a random initialization for the shape parameters
    % Values are chosen to be in a reasonable range
    center = dlarray(rand(2, num_shapes), 'CB'); 
    dimensions = dlarray(0.1 + rand(2, num_shapes) * 0.5, 'CB');
    color = dlarray(rand(3, num_shapes), 'CB');

    case "circles"
    % Make a random initialization for the shape parameters
    % Values are chosen to be in a reasonable range.

    % We set the two circles positions such that one will appear somewhere in the left side of
    % the image and one somewhere on the right side.

    center = dlarray([0.25, 0.25; 0, 0.5] + 0.5*rand(2, num_shapes), 'CB'); 
    dimensions = dlarray(0.5 + rand(2, num_shapes) * 0.5, 'CB'); 
    color = dlarray(rand(3, num_shapes), 'CB'); 
end

% Because the shapes colors are added to one another, the final
% render can have pixel values much greater than 1. 
% We normalize the initial colours to ensure the rendered image has the same 
% mean as the target image to help the optimization converge.
rendered_image = render(center, dimensions, color, X, Y, shape_type);
color = color .* (mean(target_image(:))/mean(rendered_image(:)));

% Plot the initialization render alongside the target image
figure
rendered_image = render(center, dimensions, color, X, Y, shape_type);

subplot(1, 2, 1);
imshow(extractdata(rendered_image));
title('Initialization');

subplot(1, 2, 2);
imshow(extractdata(target_image))
title('Target Image');

%% Optimize

% Set the optimization hyperparameters

% Set the learning rate. We use a slower learning rate for the circles
% example to more clearly visualize how it optimizes over time
switch example
    case {"rectangles", "gaussians"}
        learning_rate = 0.75;
    case "circles"
        learning_rate = 0.05;
end

% Multiplier for the learning rate to reduce it over time
decay_rate = 0.99;

% Total number of iterations
num_iterations = 125;

% Variables to store the loss function value over time, and the momentum
% vectors used by the sgdmupdate function 
losses = zeros(num_iterations,1);
v_center = [];
v_dimensions = [];
v_color = [];

figure

for iter = 1:num_iterations
    % Compute gradients and loss value
    [gradients, loss_value] = dlfeval(@computeGradients, center, dimensions, color, target_image, X, Y, shape_type);

    % Extract the gradients
    grad_center = gradients{1};
    grad_dimensions = gradients{2};
    grad_color = gradients{3};

    % Update shape parameters using the sgdm optimizer
    [center, v_center] = sgdmupdate(center, grad_center, v_center, learning_rate);
    [dimensions, v_dimensions] = sgdmupdate(dimensions, grad_dimensions, v_dimensions, learning_rate);
    [color, v_color] = sgdmupdate(color, grad_color, v_color, learning_rate);

    % Clip the parameters to ensure they stay within reasonable bounds
    center = clip(center, 0.01, 0.99);
    dimensions = clip(dimensions, 0.01, 1);

    % Reduce the learning rate over time to make smaller adjustments at
    % later iterations
    learning_rate = learning_rate * decay_rate;

    % Store the cost function
    losses(iter) = extractdata(loss_value);

    % Plot the rendered image alongside the target image
    rendered_image = render(center, dimensions, color, X, Y, shape_type);
    
    subplot(2, 2, 1);
    imshow(extractdata(rendered_image));
    title(['Iteration: ', num2str(iter)]);
    
    subplot(2, 2, 2);
    imshow(extractdata(target_image));
    title('Target Image');
    
    subplot(2, 2, [3, 4]);
    plot(losses);
    xlabel('Iteration');
    ylabel('Loss');
    title("Learning rate " + string(learning_rate)  + "  Loss: "+string(extractdata(loss_value)));

    drawnow;
end

%% Supporting functions

% Loss function
function loss = computeLoss(center, dimensions, color, target_image, X, Y,  shape_type)
    % Render the image given the current shape parameters, and then
    % calculate the mean-square-error of the render compared to the target
    % image.
    rendered_image = render(center, dimensions, color, X, Y, shape_type);
    loss = mean((rendered_image(:) - target_image(:)).^2);
end

%% 

% Function to compute gradients with respect to shape parameters
function [gradients, loss_value] = computeGradients(center, dimensions, color, target_image, X, Y, shape_type)
    % Evaluate the loss function and the gradient with respect to the shape
    % parameters
    loss_value = computeLoss(center, dimensions, color, target_image, X, Y, shape_type);
    gradients = dlgradient(loss_value, {center, dimensions, color});
end

%% 

% The differentiable rendering function
% 
% The edges of the shapes are blurred using a sigmoid function which makes the 
% rasterization continuous and differentiable. Different shapes are defined using 
% different signed distance functions. 
function rendered_image = render(center, dimensions, color, X, Y, shape_type)
    % This function takes the shape parameters, pixel grids, and shape type
    % and then outputs the final rendered image.

    % It uses the same process as detailed in the 'Differentiable rendering introduce'
    % section, but the code is vectorized to perform all the steps for all
    % shapes simultaneously for speed.

    switch shape_type
    case "rectangle"
        % Render each rectangle using the signed distance function followed
        % by the sigmoid function. 
        dist_to_center = max(abs(X - reshape(center(1,:), 1, 1,1, [])) ./ (0.5 * reshape(dimensions(1,:), 1, 1,1, [])), ...
                         abs(Y - reshape(center(2,:), 1, 1,1, [])) ./ (0.5 * reshape(dimensions(2,:), 1, 1,1, [])));
        mask = stableSigmoid(dist_to_center);
    case "circle"
        % Render each circle using the signed distance function followed
        % by the sigmoid function
        dist_to_center =  (X - reshape(center(1,:), 1, 1,1, [])).^2 ./ ((0.5 * reshape(dimensions(1,:), 1, 1,1, [])).^2 + 1e-8) ...
                        + (Y - reshape(center(2,:), 1, 1,1, [])).^2 ./ ((0.5 * reshape(dimensions(2,:), 1, 1,1, [])).^2 + 1e-8);
        mask = stableSigmoid(dist_to_center);
    case "gaussians"
        % Render each gaussian. This does not require signed distance
        % functions or sigmoids -- we just use the standard formula for a
        % 2D Gaussian.
        mask = exp(-((X - reshape(center(1,:), 1, 1,1, [])).^2 ./ (0.02 * reshape(dimensions(1,:), 1, 1,1, [])) + ...
                               (Y - reshape(center(2,:), 1, 1,1, [])).^2 ./ (0.02 * reshape(dimensions(2,:), 1, 1,1, []))));
    end

    % Combine each of the shapes into one image with additive blending
    % Different blending types will give different results!
    rendered_image = sum(mask .* reshape(color, 1, 1, size(color, 1), []), 4);
end

%% 

% Stable sigmoid function for smooth edge approximation
function y = stableSigmoid(x)
    y = 1 ./ (1 + exp(-clip(150 * (1-x),-20,20)));
    y = clip(y, 1e-6, 1-1e-6);
end
