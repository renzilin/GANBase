import argparse
import logging
from pathlib import Path
import sys
import time

import grpc
import numpy as np

from read_until import AccumulatingCache, ReadUntilClient

try:
    from pyguppy_client_lib.pyclient import PyGuppyClient
    from pyguppy_client_lib.helper_functions import package_read
except ImportError:
    print(
        "Failed to import pyguppy_client_lib, do you need to `pip install ont-pyguppy-client-lib`",
        file=sys.stderr,
    )

import csv
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class label2int:
    def __init__(self, baseseq = 'ACGT'):
        self.int_map = {}
        self.base_map = {}
        for ind, base in enumerate(baseseq):
            self.int_map[base] = ind
            self.base_map[ind] = base
    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            ch = self.int_map[c]
            int_sequence.append(ch)
        return int_sequence

convert = label2int()

def data_convert(sequence):
    predict_data = convert.text_to_int(sequence[:200])
    return torch.tensor([predict_data])

pe = torch.zeros(200, 4)
position = torch.arange(0, 200).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 4, 2) *  -(math.log(10000.0) / 4))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)


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
    
def prob_to_prediction(output): 
    prob= torch.exp(output)
    if prob > 0.1:
        return 1
    return 0
    

def model_load(path):
    discriminator = Discriminator().cuda()
    dis_dict = torch.load(path)
    discriminator.load_state_dict(dis_dict, False)    
    return discriminator


def basecall(
    guppy_client: PyGuppyClient, reads: list, dtype: "np.dtype", daq_values: dict,
):
    """Generator that sends and receives data from guppy

    :param guppy_client: pyguppy_client_lib.pyclient.PyGuppyClient
    :param reads: List of reads from read_until
    :type reads: Iterable
    :param dtype:
    :param daq_values:

    :returns:
        - read_info (:py:class:`tuple`) - channel (int), read number (int)
        - read_data (:py:class:`dict`) - Data returned from Guppy
    :rtype: Iterator[tuple[tuple, dict]]
    """
    hold = {}
    missing_count = 0

    with guppy_client:
        for channel, read in reads:
            hold[read.id] = (channel, read.number)
            t0 = time.time()
            success = guppy_client.pass_read(
                package_read(
                    read_id=read.id,
                    raw_data=np.frombuffer(read.raw_data, dtype),
                    daq_offset=daq_values[channel].offset,
                    daq_scaling=daq_values[channel].scaling,
                )
            )
            if not success:
                logging.warning("Skipped a read: {}".format(read.id))
                hold.pop(read.id)
                continue
            else:
                missing_count += 1

            sleep_time = guppy_client.throttle - t0
            if sleep_time > 0:
                time.sleep(sleep_time)

        while missing_count:
            results = guppy_client.get_completed_reads()
            missing_count -= len(results)

            if not results:
                time.sleep(guppy_client.throttle)
                continue

            yield from iter(
                [(hold[read["metadata"]["read_id"]], read) for read in results]
            )


def get_parser():
    """Build argument parser for example"""
    parser = argparse.ArgumentParser(prog="demo ({})".format(__file__),)
    parser.add_argument(
        "--model_path", required=True, type=str, metavar='PATH', help="Your model files paths"
    )
    parser.add_argument(
        "--barcodes", nargs="*", default=[], help="Barcode kits in use"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="MinKNOW server host address"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="MinKNOW gRPC server port"
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=None,
        help="Path to alternate CA certificate for connecting to MinKNOW.",
    )

    parser.add_argument(
        "--guppy_host", default="127.0.0.1", help="Guppy server host address",
    )
    parser.add_argument(
        "--guppy_port", type=int, default=5555, help="Guppy server port",
    )
    parser.add_argument(
        "--guppy_config", default="dna_r9.4.1_450bps_fast", help="Guppy server config",
    )

    parser.add_argument(
        "--run_time", type=int, default=30, help="Period to run the analysis"
    )
    parser.add_argument(
        "--unblock_duration",
        type=float,
        default=0.1,
        help="Time (in seconds) to apply unblock voltage",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Number of reads to get from ReadCache each iteration. If not set uses number of channels on device",
    )
    parser.add_argument(
        "--throttle",
        default=0.1,
        type=float,
        help="Time to wait between requesting successive read batches from the ReadCache",
    )
    return parser


def analysis(
    model: Discriminator,
    client: ReadUntilClient,
    caller: PyGuppyClient,
    batch_size: int,
    duration: int,
    throttle: float = 0.1,
    unblock_duration: float = 0.1,
):
    """Example analysis function

    This is an example analysis function that collects data (read chunks)
    from the `ReadUntilClient`, passes them to the `PyGuppyClient` for
    calling, aligning, or barcoding and iterates the results.

    :param client: an instance of a `ReadUntilClient` object.
    :param caller: PyGuppyClient
    :param batch_size: number of reads to pull from `client` at a time.
    :param duration: time to run for, seconds
    :param throttle: minimum interval between requests to `client`.
    :param unblock_duration: time in seconds to apply unblock voltage.
    """
    run_duration = time.time() + duration
    logger = logging.getLogger("Analysis")

    sampling_file = open('adaptive_sampling.csv', mode='w', newline='')
    sampling_writer = csv.writer(sampling_file)
    sampling_writer.writerow(['batch_time', 'read_number', 'channel', 'read_id', 'sequence_length', 'decision', 'sequence'])
    target_counter, non_target_counter, short_counter, control_counter = 0, 0, 0, 0

    while client.is_running and time.time() < run_duration:
        time_begin = time.time()
        # Get most recent read chunks from read until client
        read_batch = client.get_read_chunks(batch_size=batch_size, last=True)

        # Send read_batch for base calling
        called_batch = basecall(
            guppy_client=caller,
            reads=read_batch,
            dtype=client.signal_dtype,
            daq_values=client.calibration_values,
        )

        n = 0
        for (channel, read_number), read in called_batch:
            n += 1
            sequence = read['datasets']['sequence']
            sequence_length = len(sequence)
            read_id = read["metadata"]["read_id"]

            # 257-512 channels as control group
            if channel > 256:
                control_counter += 1
                client.stop_receiving_read(channel, read_number)
                row = [time_begin, read_number, channel, read_id, sequence_length, 'control', sequence]
                sampling_writer.writerow(row)
                continue

            if sequence_length < 200:
                short_counter += 1
                row = [time_begin, read_number, channel, read_id, sequence_length, 'too_short', sequence]
                sampling_writer.writerow(row)
                continue

            input = data_convert(sequence)
            output = model(input.cuda())
            result = prob_to_prediction(output)
            result = 1 - result.item()
            
            if result == 1:
                target_counter += 1
                client.stop_receiving_read(channel, read_number)
                row = [time_begin, read_number, channel, read_id, sequence_length, 'stop_receiving', sequence]
                sampling_writer.writerow(row)
            else:
                non_target_counter += 1
                client.unblock_read(channel, read_number, unblock_duration)
                row = [time_begin, read_number, channel, read_id, sequence_length, 'unblock', sequence]
                sampling_writer.writerow(row)

        batch_time = time.time() - time_begin
        if n:
            info_str = "batch time: {:.5f}, called batch: {}, target: {}, non-target: {}, short reads: {}, control group: {}".format(
                    batch_time, n, target_counter, non_target_counter, short_counter, control_counter)
            logger.info(info_str)
            logger.info("Processed {} in {:.5f}s".format(n, batch_time))
            print(info_str)

        # Limit the rate at which we make requests
        if batch_time < throttle:
            time.sleep(throttle - batch_time)


def main(argv=None):
    """simple example main cli entrypoint"""
    args = get_parser().parse_args(argv)

    logging.basicConfig(
        format="[%(asctime)s - %(name)s] %(message)s", level=logging.INFO,
    )

    target_model = model_load(args.model_path)

    channel_credentials = None
    if args.ca_cert is not None:
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=args.ca_cert.read_bytes()
        )
    read_until_client = ReadUntilClient(
        mk_host=args.host,
        mk_port=args.port,
        mk_credentials=channel_credentials,
        cache_type=AccumulatingCache,
        one_chunk=False,
        filter_strands=True,
        # Request uncalibrated, int16, signal
        calibrated_signal=False,
    )

    # Handle arg cases:
    if args.batch_size is None:
        args.batch_size = read_until_client.channel_count

    caller = PyGuppyClient(
        address="{}:{}".format(args.guppy_host, args.guppy_port),
        config=args.guppy_config,
        barcode_kits=args.barcodes
    )

    caller.connect()
    read_until_client.run()

    try:
        analysis(
            model=target_model,
            length=args.length,
            option=args.option,
            client=read_until_client,
            caller=caller,
            duration=args.run_time,
            batch_size=args.batch_size,
            unblock_duration=args.unblock_duration,
        )
    except KeyboardInterrupt:
        pass
    finally:
        read_until_client.reset()
        caller.disconnect()


if __name__ == "__main__":
    main()