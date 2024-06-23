import math

# Example data
O11 = 150  # Frequency of "machine learning" in target corpus
O12 = 100000 - 150  # Total phrases in target corpus excluding "machine learning"
O21 = 800  # Frequency of "machine learning" in reference corpus
O22 = 1000000 - 800  # Total phrases in reference corpus excluding "machine learning"

# Calculate the total number of phrase occurrences
N = O11 + O12 + O21 + O22

# Calculate expected frequencies
E11 = (O11 + O12) * (O11 + O21) / N
E12 = (O11 + O12) * (O12 + O22) / N
E21 = (O21 + O22) * (O11 + O21) / N
E22 = (O21 + O22) * (O12 + O22) / N

# Function to calculate the log likelihood ratio component
def LLR_component(O, E):
    return O * math.log(O / E) if O != 0 else 0

# Calculate the log likelihood ratio
LLR = 2 * (LLR_component(O11, E11) + LLR_component(O12, E12) +
           LLR_component(O21, E21) + LLR_component(O22, E22))

print(f"Log Likelihood Ratio (LLR): {LLR:.2f}")
