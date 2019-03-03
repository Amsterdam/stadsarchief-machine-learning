def decypher_line(line):
    line = line.strip()
    values = line.split(' ')

    obj_idx = int(values[0])
    bbox = [float(n) for n in values[1:]]
    return [obj_idx, bbox]
