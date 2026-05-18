import argparse
import functools
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
import typing
import queue

import grpc
import numpy as np
import math
import sys
# sys.path.append("/home/master/Desktop/GANBase/read_until_api-3.0.0")
import read_until

import os
import csv
import torch
from torch import nn
from torch.autograd import Variable

from threading import Thread
from koi.decode import beam_search, to_str
from utils import chunk, stitch, batchify, unbatchify, half_supported,load_basecall_model


pe = torch.zeros(200, 4)
position = torch.arange(0, 200).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 4, 2) *  -(math.log(10000.0) / 4))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)
__default_norm_params__ = {'quantile_a' : 0.2,
                           'quantile_b' : 0.9,
                           'shift_multiplier' : 0.51,
                           'scale_multiplier' : 0.53}

def normalisation(sig, scaling_strategy=None, norm_params=None):

    # print('norm--')
    """
    Calculate signal shift and scale factors for normalisation or standardisation.
    If no information is provided in the config, quantile scaling is default.
    """
    if scaling_strategy and scaling_strategy.get("strategy") == "pa":
        if norm_params.get("standardise") == 1:
            shift = norm_params.get('mean')
            scale = norm_params.get('stdev')
        elif norm_params.get("standardise") == 0:
            shift = 0.0
            scale = 1.0
        else:
            raise ValueError("Picoampere scaling requested, but standardisation flag not provided")

    elif scaling_strategy is None or scaling_strategy.get("strategy") == "quantile":
        if norm_params is None:
            norm_params = __default_norm_params__

        qa, qb = np.quantile(sig, [norm_params['quantile_a'], norm_params['quantile_b']])
        shift = max(10, norm_params['shift_multiplier'] * (qa + qb))
        scale = max(1.0, norm_params['scale_multiplier'] * (qb - qa))
    else:
        raise ValueError(f"Scaling strategy {scaling_strategy.get('strategy')} not supported; choose quantile or pa.")
    signal= (sig - shift) / scale
    return signal

class label2int:
	def __init__(self, baseseq = 'ACGT'):
		self.int_map = {}
		for ind, base in enumerate(baseseq):
			self.int_map[base] = ind
	def text_to_int(self, text):
		""" Use a character map and convert text to an integer sequence """
		int_sequence = []
		for c in text:
			ch = self.int_map[c]
			int_sequence.append(ch)
		return int_sequence

convert = label2int()

def read_encode(sequences):
    predict_data = torch.tensor(convert.text_to_int(sequences) )
    return predict_data


class PositionalEncoding(nn.Module):
	def __init__(self, pe):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=0.2)
		self.register_buffer('pe', pe)
	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
		return self.dropout(x)


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.embed = nn.Embedding(4, 4)
		self.pe = PositionalEncoding(pe)
		self.ln = nn.Linear(4, 64)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead = 2, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
		self.conv1 = nn.Sequential(
					nn.Conv1d(64, 128, 5, 1, 2),  
					nn.ReLU(),
				)
		self.conv2 = nn.Sequential(
					nn.Conv1d(128, 256, 5, 1, 2), 
					nn.ReLU(),
				)
		self.conv3 = nn.Sequential(
					nn.Conv1d(256, 256, 5, 1, 2),  
					nn.ReLU(),
				)
		self.normlayer = nn.LayerNorm(256)
		self.fc = nn.Linear(51200 , 2)
		self.activation = nn.LogSoftmax(dim=1)
	def forward(self,x):
		x = self.embed(x)  
		se = x        
		x = self.pe(x)  
		x = x + se
		x = self.ln(x)
		se = x
		x = self.transformer_encoder(x)        
		x = x + se
		x = x.permute(0,2,1)  # batch_size, hidden_dim, seq_len
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.permute(0,2,1)  # batch_size, hidden_dim, seq_len
		x = self.normlayer(x)
		x = self.normlayer(x)
		x =  x.contiguous().view((x.size()[0], -1)) # batch_size, seq_len*hidden_dim*2
		x = self.fc(x)
		x = self.activation(x)
		return x

def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    if length < size:
        return results[0, :int(np.floor(length / stride))]
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model.
    """
    with torch.inference_mode():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        with torch.cuda.device(scores.device):
            sequence, qstring, moves = beam_search(
                scores, beam_width=beam_width, beam_cut=beam_cut,
                scale=scale, offset=offset, blank_score=blank_score
            )
        return {
            'sequence': sequence,
        }


class BackgroundIterator:
    """
    Runs an iterator in the background.
    """
    def __init__(self, iterator, maxsize=10):
        super().__init__()
        self.iterator = iterator
        self.queue = self.QueueClass(maxsize)

    def __iter__(self):
        self.start()
        while True:
            item = self.queue.get()
            if item is StopIteration:
                break
            yield item

    def run(self):
        for item in self.iterator:
            self.queue.put(item)
        self.queue.put(StopIteration)

    def stop(self):
        self.join()

class ThreadIterator(BackgroundIterator, Thread):
    """
    Runs an iterator in a separate process.
    """
    QueueClass = queue.Queue

def thread_iter(iterator, maxsize=1):
    """
    Take an iterator and run it on another thread.
    """
    return iter(ThreadIterator(iterator, maxsize=maxsize))


def basecall(model, reads, raw_data, chunksize=4000, overlap=100, batchsize=32,
             reverse=False, rna=False):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, raw_data[ind].shape[-1]), chunk(torch.from_numpy(normalisation(raw_data[ind])[200:]), chunksize, overlap))
        for ind,read in enumerate(reads)
    ) 
    batches = thread_iter(batchify(chunks, batchsize=batchsize))
    scores = thread_iter(
        (read, compute_scores(model, batch, reverse=reverse)) for read, batch in batches
    )
    results = thread_iter(
        (stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((_, start, end), scores) in unbatchify(scores)
    )
    result_lst = list(
        to_str(attrs['sequence'])
        for attrs in results)
    return result_lst

def basecall_func(pred_signal,raw_data):
    in_basecall_time= time.time()
    basecall_results = basecall(
        basecall_model, pred_signal,raw_data,
        batchsize=basecall_model.config["basecaller"]["batchsize"],
        chunksize=basecall_model.config["basecaller"]["chunksize"],
        overlap=basecall_model.config["basecaller"]["overlap"],
    )
    # print('in_basecall_time',time.time()-in_basecall_time)
    return basecall_results


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for example"""
    parser = argparse.ArgumentParser("Read until API demonstration..")
    parser.add_argument("--host", default="127.0.0.1", help="MinKNOW server host.")
    parser.add_argument(
        "--port", type=int, default=8000, help="MinKNOW gRPC server port."
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=None,
        help="Path to alternate CA certificate for connecting to MinKNOW.",
    )
    parser.add_argument("--workers", default=1, type=int, help="worker threads.")
    parser.add_argument(
        "--analysis_delay",
        type=int,
        default=1,
        help="Period to wait before starting analysis.",
    )
    parser.add_argument(
        "--run_time", type=int, default=14400, help="Period to run the analysis."
    )
    parser.add_argument(
        "--unblock_duration",
        type=float,
        default=0.1,
        help="Time (in seconds) to apply unblock voltage.",
    )
    parser.add_argument(
        "--one_chunk",
        default=False,
        action="store_true",
        help="Minimum read chunk size to receive.",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=2200,
        # default=1000,
        help="Minimum read chunk size to receive. NOTE: this functionality "
        "is currently disabled; read chunks received will be unfiltered.",
    )
    parser.add_argument(
        "--debug",
        help="Print all debugging information",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Print verbose messaging.",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
    )
    return parser


def simple_analysis(
    classificaiton_model,
    basecall_model,
    device,
    client: read_until.ReadUntilClient,
    batch_size: int = 512,
    delay: float = 0.85,
    throttle: float = 0.1,
    unblock_duration: float = 0.1,
):
    """A simple demo analysis leveraging a `ReadUntilClient` to manage
    queuing and expiry of read data.

    :param client: an instance of a `ReadUntilClient` object.
    :param batch_size: number of reads to pull from `client` at a time.
    :param delay: number of seconds to wait before starting analysis.
    :param throttle: minimum interval between requests to `client`.
    :param unblock_duration: time in seconds to apply unblock voltage.

    """
    logger = logging.getLogger("Analysis")
    logger.warning(
        "Initialising simple analysis. "
        "This will likely not achieve anything useful. "
        "Enable --verbose or --debug logging to see more."
    )

    # we sleep a little simply to ensure the client has started initialised
    logger.info("Starting analysis of reads in %ss.", delay)
    time.sleep(delay)

    sampling_file = open('/home/master/Desktop/GANBase/out/GANBase_adaptive_sampling.csv', mode='w', newline='')
    sampling_writer = csv.writer(sampling_file)
    sampling_writer.writerow(['batch_time', 'read_number', 'channel', 'read_id', 'samples_length', 'decision'])

    time_file = open('/home/master/Desktop/GANBase/out/GANBase_adaptive_sampling_time.csv', mode='w', newline='')
    time_writer = csv.writer(time_file)
    time_writer.writerow(['batch_time', 'batch_size', 'get_read_time', 'control_time', 'signal_time', 'basecall_time', 'short_time', 'model_infer_time', 'classify_infer_time'])

    target_counter, non_target_counter, short_counter, control_counter = 0, 0, 0, 0
    # print("target_counter, non_target_counter, short_counter, control_counter",target_counter, non_target_counter, short_counter, control_counter)

    while client.is_running:
        time_begin = time.time()
        read_batch = client.get_read_chunks(batch_size=batch_size, last=True)
      
        get_read_time = time.time() - time_begin
        
        predict_lst=[]
        
        ## control filter
        control_time_begin = time.time()
        control_lst = [(channel, read) for channel, read in read_batch if channel > 256]
        for channel, read in control_lst:
            control_counter += 1
            client.stop_receiving_read(channel, read.number)
            # print("control stop_receiving_read")
            row = [time_begin, read.number, channel, read.id, len(np.frombuffer(read.raw_data, client.signal_dtype)), 'control']
            sampling_writer.writerow(row)
        control_time = time.time() -control_time_begin

        signal_time_begin = time.time()
        predict_lst = [(channel, read, np.frombuffer(read.raw_data, client.signal_dtype))
                        for channel, read in read_batch if channel <=256 and len(np.frombuffer(read.raw_data, client.signal_dtype)) >= 3000 ]
        # print('predict_lst',len(predict_lst))
        if len(predict_lst)==0:
            continue
        # print('predict_lst',len(predict_lst))
        pred_channel, pred_read_cls, raw_data = zip(*(predict_lst))
        #print("pred_read_cls",pred_read_cls)
        signal_time = time.time()-signal_time_begin

        basecall_time_begin = time.time()
        basecalled_reads = basecall_func(pred_read_cls,raw_data)
        basecall_time = time.time() -basecall_time_begin
        # print('basecall_time', basecall_time)

        if len(basecalled_reads) ==0:
            continue 

        short_time_begin  = time.time()
        
        unblock_reads = [(pred_channel[idx], pred_read_cls[idx]) for idx, read in enumerate(basecalled_reads) if len(read) < 250]
        
        for channel, read in unblock_reads:
            short_counter += 1
            # print("short_counter",short_counter)
            client.unblock_read(channel, read.number)
            # client.unblock_read(channel, read.id)
            row = [time_begin, read.number, channel, read.id, len(read.raw_data), 'short']
            # row = [time_begin, channel, read.id, len(read.raw_data), 'short']
            sampling_writer.writerow(row)
        short_time = time.time() -short_time_begin  
        # print('short_time', short_time)

        data_time_begin = time.time()
        
      
        
        
        #read_list = [(pred_channel[idx], pred_read_cls[idx], np.frombuffer(read.raw_data, client.signal_dtype)) for idx, read in enumerate(basecalled_reads) if len(read) >= 250 ]
        
        read_list = [(pred_channel[idx], pred_read_cls[idx], read) for idx, read in enumerate(basecalled_reads) if len(read) >= 250 ]
        
        
        
        if_save_reads = [ read_encode( basecalled_reads[idx][-200:] ) for idx, read in enumerate(basecalled_reads) if len(read) >= 250 ]
        
        if len(if_save_reads) ==0 :
             continue
        output_label  = torch.exp(classificaiton_model(torch.stack(if_save_reads).cuda() ))
        model_infer_time = time.time() -data_time_begin
        # print('model_infer_time', model_infer_time)

        classify_infer_time_begin = time.time()
        for idx, label in enumerate(output_label):
            # print('label',label[1])  
            channel, read , read_raw_data = read_list[idx]
            if label[1] > 0.9:
                ## human filter
                client.unblock_read(channel, read.number)
                row = [time_begin, read.number, channel, read.id, len(read_raw_data), 'unblock']
                #client.unblock_read(channel, read.id)
                #row = [time_begin, read.id, channel, read.id, len(read_raw_data), 'unblock']
                sampling_writer.writerow(row)
                non_target_counter += 1
            else:
                ## zymo filter
                client.stop_receiving_read(channel, read.number)
                row = [time_begin, read.number, channel, read.id, len(read_raw_data), 'stop_receiving']
                #client.stop_receiving_read(channel, read.id)
                #row = [time_begin, read.id, channel, read.id, len(read_raw_data), 'stop_receiving']
                sampling_writer.writerow(row)
                target_counter += 1
        classify_infer_time = time.time() -classify_infer_time_begin
        
        # limit the rate at which we make requests
        time_end = time.time()
        if time_begin + throttle > time_end:
            time.sleep(throttle + time_begin - time_end)

        time_row = [time_end-time_begin, len(read_batch), get_read_time, control_time, signal_time, basecall_time, short_time, model_infer_time, classify_infer_time]
        time_writer.writerow(time_row)
        # print("target_counter, non_target_counter, short_counter, control_counter",target_counter, non_target_counter, short_counter, control_counter)
        print("batch time: {}, batch size: {}, target reads: {}, non-target reads: {}, short reads: {}, control group reads: {}".format(
            time_end-time_begin, len(read_batch), target_counter, non_target_counter, short_counter, control_counter))
        print("get_read_time,{},control_time: {}, signal_time: {}, basecall_time: {},short_time:{}, model_infer_time: {}, classify_infer_time: {}".format(
            get_read_time, control_time, signal_time, basecall_time, short_time, model_infer_time, classify_infer_time))
    return target_counter


def run_workflow(
    client: read_until.ReadUntilClient,
    analysis_worker: typing.Callable[[], None],
    n_workers: int,
    run_time: float,
    runner_kwargs: typing.Optional[typing.Dict] = None,
):
    """Run an analysis function against a ReadUntilClient.

    :param client: `ReadUntilClient` instance.
    :param analysis worker: a function to process reads. It should exit in
        response to `client.is_running == False`.
    :param n_workers: number of incarnations of `analysis_worker` to run.
    :param run_time: time (in seconds) to run workflow.
    :param runner_kwargs: keyword arguments for `client.run()`.

    :returns: a list of results, on item per worker.

    """
    print("run workflow.")
    logger = logging.getLogger("Manager")
    if not runner_kwargs:
        runner_kwargs = {}

    results = []
    pool = ThreadPool(n_workers)
    logger.info("Creating %s workers", n_workers)
    try:

        # start the client
        client.run(**runner_kwargs)

        # start a pool of workers
        for _ in range(n_workers):
            results.append(pool.apply_async(analysis_worker))
        pool.close()

        # wait a bit before closing down
        time.sleep(run_time)
        logger.info("Sending reset")
        client.reset()
        pool.join()

    except KeyboardInterrupt:
        logger.info("Caught ctrl-c, terminating workflow.")
        client.reset()

    # collect results (if any)
    collected = []

    for result in results:

        try:
            res = result.get(3)
        except TimeoutError:
            logger.warning("Worker function did not exit successfully.")
            collected.append(None)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Worker raise exception:")
        else:
            logger.info("Worker exited successfully.")
            collected.append(res)
    pool.terminate()
    return collected


def main(argv=None):
    """simple example main cli entrypoint"""
    print('START...')
    args = get_parser().parse_args(argv)

    logging.basicConfig(
        format="[%(asctime)s - %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=args.log_level,
    )

    channel_credentials = None
    if args.ca_cert is not None:
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=args.ca_cert.read_bytes()
        )
    # channel = grpc.insecure_channel("127.0.0.1:8000")
    read_until_client = read_until.ReadUntilClient(
        mk_host=args.host,
        mk_port=args.port,
        mk_credentials=channel_credentials,
        one_chunk=args.one_chunk,
        filter_strands=True,
    )
    print("read_until_client finished.")

    analysis_worker = functools.partial(
        simple_analysis,
        classificaiton_model,
        basecall_model,
        device,
        read_until_client,
        delay=args.analysis_delay,
        unblock_duration=args.unblock_duration,
    )
    print("analysis_worker.")

    results = run_workflow(
        read_until_client,
        analysis_worker,
        args.workers,
        args.run_time,
        runner_kwargs={"min_chunk_size": args.min_chunk_size},
    )
    print("done result")
    
    
    
    for idx, result in enumerate(results):
        print("Worker " ,(idx + 1)," received", result, " target reads", )
        logging.info("Worker %s received %s target reads", idx + 1, result)


if __name__ == '__main__':
    def init(device, deterministic=True):
        if device == "cpu": return
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = (not deterministic)
        assert(torch.cuda.is_available())
    
    device = 'cuda'
    init(device)

    classificaiton_model_path = '/home/master/Desktop/GANBase/model_dis10.pt'
    classificaiton_model = Discriminator().cuda()
    dis_dict = torch.load(classificaiton_model_path)
    classificaiton_model.load_state_dict(dis_dict, False)  
    

    basecall_model_directory ='/home/master/Desktop/GANBase/bonito-0.9.1/bonito/models/dna_r10.4.1_e8.2_400bps_hac@v5.2.0'

    basecall_model = load_basecall_model(
            basecall_model_directory,
            device,
            quantize=None,
            use_koi=True,)
    print('load model finish.')

    main()
