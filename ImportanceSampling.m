% Importance Sampling Algorithm

% Seed for consistent results.
rng(10);

% Proposal Distribution Parameters.
% Mean.
mu = 0;

% Standard Deviation - to obtain dispayed graphs modify to 0.5,1,5,10.
sigma = 5;

% Importance Sampling Functions.
% Function to calculate f(x) for a given sample from the proposal
f = @(x) 0.2 *( x + 5.0 );

% Function to define the Target Distribution p(x)
p = @(x) exp((0.4.*(x-0.4).^2) - (0.08.*(x.^4)));

% Proposal distribution q(x) (assume unnormalized)
q = @(x)  1/(sqrt(2*pi)*sigma) * exp(-(x-mu).^2/(2*sigma^2));
 
% Empty vector to store weights.
weights = [];

% Emply vector to store E[x] at each iteration.
f_estimate = {};

% Generate vector for N samples - log scale.
tens = ones([1,10]);
hundreds = ones([1,10]);
thousands = ones([1,10]); 
tenthoudand = ones([1,10]);
hundredthousand = ones([1,10]);
millions = ones([1,10]);
iterations = [];

for i = 1:9
    tens(i) = tens(i)*10*i;
    hundreds(i) = hundreds(i)*100*i;
    thousands(i) = thousands(i)*1000*i ;
    tenthoudand(i) = tenthoudand(i)*10000*i ;
    hundredthousand(i) = hundredthousand(i)*100000*i ;
    millions(i) = millions(i)*1000000*i;
end
% Remove the last elements in the vector.
tens(end) = [];
hundreds(end) = [];
thousands(end) = [];
tenthoudand(end) = [];
hundredthousand(end) =[];
millions(end) = []; 
iterations = [1, tens, hundreds, thousands, tenthoudand,hundredthousand, millions, 1e7];

% Importance Sampling.
for i = 1:length(iterations)
        
    % Randomly sample from the proposal distribution.
    x = normrnd(mu,sigma,iterations(i),1);
    % Compute weights.
    weights = p(x)./q(x);
    % Calculate the estimate value E[x]
    e_x = (weights/sum(weights)).*f(x);
    % Assign the sum of the estimated values
    f_estimate{i}  = sum(e_x);
    % Numerator of the estimate.
    E_num = weights.*f(x);
 end

% Calculate the estimate E[x] convergence value.
estimate = sum(E_num)/sum(weights);
% Empty vector to store mean estimate values.
f_estimate_mean = zeros(size(f_estimate,2),1);

% Calculate the mean E[f(x)] estimates at each iteration.
for i=1:size(f_estimate,2)
    f_estimate_mean(i) = mean([f_estimate{i}]);
end
 
% Plot importance Sampling.
% Plot of the average E[f(x)]
semilogx(iterations, f_estimate_mean,'-or', 'MarkerFaceColor', 'r'); hold on;
% Plot the estimated convergence value.
semilogx(iterations, estimate*ones(size(iterations)),'k'); hold on;
xlabel('number of iterations'); ylabel('E[f(x)] estimate');

% Generate Matrix of iteration and E[f(x)] at iteration.
format short g
results_matrix = [iterations',f_estimate_mean];
results_matrix