import argparse     # handle command line arguments
import util

if __name__ == "__main__":
    """
    Train a Detectron2 detector on a COCO-JSON dataset

    Expected layout (relative to --data-dir):
      data/
        annotations/
          instances_train.json
          instances_val.json
        train/
          imgs/
            .jpg
        val/
          imgs/
            .jpg
    Command:
    python train.py --device cuda --iterations 500
    """
    parser = argparse.ArgumentParser(description="Train Detectron2 on COCO-JSON data")

    # path arguments
    parser.add_argument('--class-list', default='./class.names')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output')

    # hardware device
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])

    # training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.00025)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--checkpoint-period', type=int, default=100)

    # model selection
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

    # parse all arguments from command line
    args = parser.parse_args()

    # call the training defined in util.py
    # this function: register dataset, config building, trainer creation,
    # validation hook setup, and starting training.
    util.train(output_dir=args.output_dir,
               data_dir=args.data_dir,
               class_list_file=args.class_list,
               learning_rate=args.learning_rate,
               batch_size=args.batch_size,
               iterations=args.iterations,
               checkpoint_period=args.checkpoint_period,
               device=args.device,
               model=args.model, )