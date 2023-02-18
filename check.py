import os
path = './exprs'

for expr_name in os.listdir(path):
    if expr_name.startswith('.'):
        continue
    expr_dir = os.path.join(path, expr_name)
    for ckpt_name in os.listdir(expr_dir):
        if ckpt_name.startswith('.'):
            continue
        ckpt_dir = os.path.join(expr_dir, ckpt_name)
        if os.path.isdir(ckpt_dir):
            
            ckpt_dir = os.path.join(ckpt_dir, 'chckpoints')
            if os.path.exists(ckpt_dir):
                iter_names = [name for name in os.listdir(ckpt_dir) if not name.startswith('.')]
                
                if len(iter_names) == 0:
                    print(ckpt_dir, None)
                else:
                    max_len = max([len(name) for name in iter_names])
                    iter_names = [name for name in iter_names if len(name) == max_len]
                    max_iter_name = max(iter_names)
                    print(ckpt_dir, max_iter_name)
            else:
                print(ckpt_dir, None)