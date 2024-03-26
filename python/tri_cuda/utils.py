def simple_ptx_info_extract(ptx_file:str):
    import re
    ptx_info = {}
    with open(ptx_file, 'r') as f:
        for line in f.readlines():
            matched_res = re.search(r'.visible .entry (.*)\(', line)
            if matched_res:
                ptx_info['kernel_name'] = str(matched_res.group(1).strip(' ').strip('\t'))
                continue
            
            matched_res = re.search(r'.maxntid (\d+), (\d+), (\d+)', line)
            if matched_res:
                ptx_info['block'] = (int(matched_res.group(1)), int(matched_res.group(2)), int(matched_res.group(3)))
                continue
            
    return ptx_info


