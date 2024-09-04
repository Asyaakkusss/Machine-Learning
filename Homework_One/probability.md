a) Given the following probabilities, compute the MAP decision rule by specifying the output
𝑦̂ (either 0 or 1) for each possible input combination. Compute also its misclassification
probability 𝑃(𝑌̂ ≠ 𝑌), which is the probability that the MAP decision rule output (pre-
dicted target) does not match the actual target.
𝑃(𝑌 = 1) = 0.2
𝑃(𝑋1 = 1, 𝑋2 = 1|𝑌 = 1) = 0.5
𝑃(𝑋1 = 1, 𝑋2 = 0|𝑌 = 1) = 0.2
𝑃(𝑋1 = 0, 𝑋2 = 1|𝑌 = 1) = 0.2
𝑃(𝑋1 = 0, 𝑋2 = 0|𝑌 = 1) = 0.1
𝑃(𝑋1 = 1, 𝑋2 = 1|𝑌 = 0) = 0.1
𝑃(𝑋1 = 1, 𝑋2 = 0|𝑌 = 0) = 0.2
𝑃(𝑋1 = 0, 𝑋2 = 1|𝑌 = 0) = 0.2
𝑃(𝑋1 = 0, 𝑋2 = 0|𝑌 = 0) = 0.5

# answer to part A
x1 = 1, x2 = 1: for y = 1 it is 0.5 * 0.2 = 0.1. for y = 0 it is 0.1 * 0.8 = 0.08. By the 
MAP decision rule, 𝑦̂ = 1.

x1 = 1, x2 = 0: for y = 1 it is 0.2 * 0.2 = 0.04. for y = 0 it is 0.2 * 0.8 = 0.16. By the 
MAP decision rule, 𝑦̂ = 0. 

x1 = 0, x2 = 1: for y = 1 it is 0.2 * 0.2 = 0.04. for y = 0 it is 0.2 * 0.8 = 0.16. By the 
MAP decision rule, 𝑦̂ = 0. 

x1 = 0, x2 = 0: for y = 1 it is 0.1 * 0.2 = 0.02. for y = 0 it is 0.5 * 0.8 = 0.4. By the 
MAP decision rule, 𝑦̂ = 0. 


b) Compute the decision rule under the naïve Bayes assumption and its misclassification prob-
ability. When computing the misclassification probability, be sure to use the actual class-
conditional probabilities 𝑃(𝑋1 = 𝑥1, 𝑋2 = 𝑥2|𝑌 = 𝑦) from part (a), not the class-conditional probabilities assumed by naïve Bayes. Use the naïve Bayes assumption only to compute the decision rule!

x1 = 1, x2 = 1: for y = 1 it is 0.7 * 0.7 * 0.2 = 0.098. for y = 0 it is 0.3 * 0.3 * 0.8 = 0.072. By the naive Bayes assumption, 𝑦̂ = 1. 

x1 = 1, x2 = 0: for y = 1 it is 0.7 * 0.3 * 0.2 = 0.042. for y = 0 it is 0.3 * 0.7 * 0.8 = 0.168. By the naive Bayes assumption, 𝑦̂ = 0. 

x1 = 0, x2 = 1: for y = 1 it is 0.3 * 0.7 * 0.2 = 0.042. for y = 0 it is 0.7 * 0.3 * 0.8 = 0.168. By the naive Bayes assumption, 𝑦̂ = 0. 

x1 = 0, x2 = 0: for y = 1 it is 0.3 * 0.3 * 0.2 = 0.018. for y = 0 it is 0.7 * 0.7 * 0.8 = 0.392. By the naive Bayes assumption, 𝑦̂ = 0. 

Compare the two decision rules and their misclassification probabilities. Is the naïve Bayes assumption a good fit for this problem?
The Naive Bayes Assumption is calculated like so: P(x1 | y) * P(x2 | y) * P(y = y). It is a good fit for this problem because the MAP and Bayes 𝑦̂ results are the same. Additionally, the misclassification probabilities are very similar or the same in each case. 

