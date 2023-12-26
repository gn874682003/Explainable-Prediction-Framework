import csv

#读取文件
def readcsv(eventlog):
    csvfile = open(eventlog, 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    header = next(spamreader)
    # next(spamreader, None)  # skip the headers
    for line in spamreader:
        sequence.append(line)
    return sequence, header

#检测是否为数值类型
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



