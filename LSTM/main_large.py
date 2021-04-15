import argparse
import logging
import os
import random

import numpy as np
import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from torchtext import data
from torchtext.data import Field
from model import LSATLSTM
from utils_multiple_choice import convert_examples_to_features, processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)



def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " #+ ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--adam_betas', default='(0.9, 0.999)', type=str, help='betas for Adam optimizer')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_clip_grad_norm", action="store_true", help="whether not to clip grad norm")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup over warmup ratios.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--test_file", type=str, default="", help="file for test")
    parser.add_argument("--train_file", type=str, default="", help="file for training")
    parser.add_argument("--dev_file", type=str, default="", help="file for development")
    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def derive_features_for_model(examples,text_field,processor,args, evaluate=False, test=False, type='ar'):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"+"_%s"%type
    elif test:
        cached_mode = "test"
        cached_mode = cached_mode + type
    else:
        cached_mode = "train"+"_%s"%type
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_lstm".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )


    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        logger.info("Training number: %s", str(len(examples)))

        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            text_field
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([fea.choices_features['input_ids'].numpy() for fea in features], dtype=torch.long)
    all_input_length = torch.tensor([fea.choices_features['lengths'].numpy() for fea in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    # print(all_input_ids.size(), all_input_mask.size(), all_segment_ids.size(), all_label_ids.size())

    dataset = TensorDataset(all_input_ids, all_input_length, all_label_ids)
    return dataset



def train(args, train_dataset, val_dataset, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        str_list = str(args.output_dir).split('/')
        tb_log_dir = os.path.join('summaries', str_list[-1])
        tb_writer = SummaryWriter(tb_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    exec('args.adam_betas = ' + args.adam_betas)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)
    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("************************* Running training *************************")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
    # logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    # val_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True, test=False)

    def evaluate_model(train_preds, train_label_ids, tb_writer, args, model, best_steps, best_dev_acc, val_dataset):
        train_preds = np.argmax(train_preds, axis=1)
        train_acc = simple_accuracy(train_preds, train_label_ids)
        train_preds = None
        train_label_ids = None
        results = evaluate(args, model, val_dataset)
        logger.info(
            "dev acc: %s, loss: %s, global steps: %s",
            str(results["eval_acc"]),
            str(results["eval_loss"]),
            str(global_step),
        )
        tb_writer.add_scalar("training/acc", train_acc, global_step)
        for key, value in results.items():
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        if results["eval_acc"] > best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("!!!!!!!!!!!!!!!!!!!! achieve BEST dev acc: %s at global step: %s",
                        str(best_dev_acc),
                        str(best_steps)
                        )

            # if args.do_test:
            #     results_test, _ = evaluate(args, model, tokenizer, test=True)
            #     for key, value in results_test.items():
            #         tb_writer.add_scalar("test_{}".format(key), value, global_step)
            #     logger.info(
            #         "test acc: %s, loss: %s, global steps: %s",
            #         str(results_test["eval_acc"]),
            #         str(results_test["eval_loss"]),
            #         str(global_step),
            #     )

            # save best dev acc model
            # output_dir = os.path.join(args.output_dir, "checkpoint-best")
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            logger.info("Current local rank %s", args.local_rank)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
            with open(txt_dir, 'w') as f:
                rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
                f.write(rs)
                tb_writer.add_text('best_results', rs, global_step)

        logger.info("current BEST dev acc: %s at global step: %s",
                    str(round(best_dev_acc, 4)),
                    str(best_steps)
                    )

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    # def save_model(args, model, tokenizer):
    #     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(output_dir)
    #     tokenizer.save_vocabulary(output_dir)
    #     tokenizer.save_pretrained(output_dir)
    #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
    #     logger.info("Saving model checkpoint to %s", output_dir)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None
    model.zero_grad()
    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for epoch_index in range(int(args.num_train_epochs)):
        logger.info('')
        logger.info('%s Epoch: %d %s', '*'*50, epoch_index, '*'*50)
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "inputs": batch[0],
                'seq_lengths':batch[1],
                "labels": batch[2],
            }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]

            # print(outputs[2][0].size())
            # print(outputs[2][1].size())

            ################# work only gpu = 1 ######################
            if train_preds is None:
                train_preds = logits.detach().cpu().numpy()
                train_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                train_preds = np.append(train_preds, logits.detach().cpu().numpy(), axis=0)
                train_label_ids = np.append(train_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            ###########################################################

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if step % 20 == 0:
                logger.info("********** Iteration %d: current loss: %s", step, str(round(loss.item(), 4)),)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, tb_writer, args, model,  best_steps, best_dev_acc, val_dataset)
                        # tb_writer.add_scalar("training/lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("training/lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("training/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logger.info(
                            "Average loss: %s, average acc: %s at global step: %s",
                            str((tr_loss - logging_loss) / args.logging_steps),
                            str(train_acc),
                            str(global_step),
                        )
                        logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     save_model(args, model, tokenizer)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

    if args.local_rank in [-1, 0] and train_preds is not None:
        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, tb_writer, args, model, best_steps, best_dev_acc, val_dataset)
        # save_model(args, model, tokenizer)
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, val_dataset=None, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = val_dataset

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("************************* Running evaluation {} *************************".format(prefix))
        logger.info("Num examples = %d", len(eval_dataset))
        logger.info("Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        logger.info("Evaluating.................")
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # for batch in eval_dataloader:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "inputs": batch[0],
                    'seq_lengths': batch[1],
                    "labels": batch[2],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        # output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        # with open(output_eval_file, "w") as writer:
        #     logger.info("***** Eval results {} *****".format(str(prefix) + "----is test:" + str(test)))
        #
        #     # if not test:
        #     for key in sorted(result.keys()):
        #         if test:
        #             logger.info("%s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("***** Eval results {} *****".format(str(prefix) + "----is test:" + str(test)))
        if test:
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(round(result[key],4)))

    if test:
        return results, preds
    else:
        return results


def main():
    args = init_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        logger.warning('local_rank: %s, gpu_num: %s', torch.distributed.get_rank(), torch.cuda.device_count(),)
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    # set random seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # logger.info('n_gpu: %s, world_size: %s', args.n_gpu, torch.distributed.get_world_size())

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()



    # model = model_class
    text_field = Field(sequential=True, tokenize='basic_english',lower=False,batch_first=True,include_lengths=True,fix_length=args.max_seq_length,use_vocab=True)
    train_examples, train_texts = processor.get_train_examples(args.data_dir,args.train_file,text_field)
    dev_examples, dev_texts = processor.get_dev_examples(args.data_dir,args.dev_file,text_field)
    text_field.build_vocab(train_texts + dev_texts,vectors='glove.840B.300d')
    train_dataset = derive_features_for_model(train_examples,text_field,processor,args,evaluate=False,type=args.train_file)
    val_dataset = derive_features_for_model(dev_examples, text_field, processor,args, evaluate=True, type=args.dev_file)
    cached_vocab_file = os.path.join(
        args.data_dir,
        'pretrained_embeddings'
    )

    model = LSATLSTM(vocab_size=len(text_field.vocab),input_size=300,hidden_size=300,batch_first=True,max_seq_length=args.max_seq_length)
    if os.path.exists(cached_vocab_file) and not args.overwrite_cache:
        logger.info("Loading embeddings from cached file %s", cached_vocab_file)
        embeddings = torch.load(cached_vocab_file)
    else:
        logger.info("Saving embeddings at file %s", cached_vocab_file)
        embeddings = text_field.vocab.vectors
        torch.save(embeddings, cached_vocab_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    model.embedding.weight.data = embeddings.cuda()
    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        global_step, tr_loss, best_steps = train(args, train_dataset,val_dataset,model)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    """
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model
        # )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
    """

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     if not args.do_train:
    #         args.output_dir = args.model_name_or_path
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #         )
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    #
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=prefix)
    #         result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    #         results.update(result)

    # Test
    results = {}
    # test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True)
    # test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_exam')

    # test_dataset_rc = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_rc')
    # test_dataset_lr = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_lr')
    '''
    test_dataset_ar = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='ar')

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            checkpoint_dir = args.model_name_or_path
        if args.evaluate_during_training:
            checkpoint_dir = os.path.join(args.output_dir)
        logger.info('load checkpoint_dir: %s', checkpoint_dir)
        logger.info('current local rank: %s', args.local_rank)
        if best_steps:
            logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)

        model = model_class.from_pretrained(checkpoint_dir)
        model.to(args.device)
        # result, preds = evaluate(args, model, tokenizer, test_dataset, test=True)
        result, preds = evaluate(args, model, tokenizer, test_dataset_ar, test=True)
        # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        # results.update(result)
        np.save(os.path.join(args.output_dir, "test_preds.npy" if args.output_dir is not None else "test_preds.npy"), preds)
        # np.save(os.path.join(args.output_dir, "test_preds_lr.npy" if args.output_dir is not None else "test_preds_lr.npy"), preds)

        # evaluate(args, model, tokenizer, test_dataset_rc, test=True)
        # evaluate(args, model, tokenizer, test_dataset_lr, test=True)
        # evaluate(args, model, tokenizer, test_dataset_ar, test=True)
    '''

    # # Test for each category
    # results = {}
    # test_dataset_rc = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_rc')
    # test_dataset_lr = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_lr')
    # test_dataset_ar = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True, type='_ar')
    # 
    # if args.do_test and args.local_rank in [-1, 0]:
    #     if not args.do_train:
    #         checkpoint_dir = args.model_name_or_path
    #     if args.evaluate_during_training:
    #         checkpoint_dir = os.path.join(args.output_dir)
    #     logger.info('load checkpoint_dir: %s', checkpoint_dir)
    #     logger.info('current local rank: %s', args.local_rank)
    #     if best_steps:
    #         logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    # 
    #     model = model_class.from_pretrained(checkpoint_dir)
    #     model.to(args.device)
    #     evaluate(args, model, tokenizer, test_dataset_rc, test=True)
    #     evaluate(args, model, tokenizer, test_dataset_lr, test=True)
    #     evaluate(args, model, tokenizer, test_dataset_ar, test=True)

    return results

if __name__ == "__main__":
    main()
