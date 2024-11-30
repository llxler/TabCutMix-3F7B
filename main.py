import torch
from utils import execute_function, get_args

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    if not args.save_path:
        args.save_path = f'synthetic/{args.dataname}/{args.method}.csv'
    main_fn = execute_function(args.method, args.mode)

    main_fn(args)

# python main.py --dataname shoppers --method tabddpm --mode train
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path fucker.csv        # 原始data
# python main.py --dataname shoppers --method tabddpm --mode sample --save_path step_-20.csv    # 采用-20cc的data