# The pattern that will be used to match the date when using regex.
DATE_PATTERN = r"(\d{4}|\d{2}|\d)[-/\s.]([A-Za-z]{3}|\d{2}|\d)[-/\s.](\d{4}|\d{2})"

# The pattern that will be used to match the total when using regex.
TOTAL_PATTERN = r"^[^\+\-]*(\d+\.(\d{2}|\d))"

# The vocabulary.
# As the vocabulary has illegal character '"' at position 2,
# we need to add an escape character at that position.
# This will allow us to use double quotes when you normally would not be allowed
VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"

# The maximum text length
TEXT_MAX_LENGTH = 68
