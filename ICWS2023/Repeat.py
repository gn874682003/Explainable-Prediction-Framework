import csv

# read file
def readcsv(eventlog):
    csvfile = open(eventlog, 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    header = next(spamreader)
    # next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    return sequence, header

# Detect whether it is a value type
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False



