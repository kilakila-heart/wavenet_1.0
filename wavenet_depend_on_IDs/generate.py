from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.3
#ID = 53    #for speaker 280
ID = 102    #for speaker 360
#ID = 16    #for speaker 243


def get_arguments():
    def _str_to_bool(s): #与快速生成相关
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,  #8000
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--ID_generation',#生成的ID
        type=int,
        default=ID,
        help='The ID to generate')
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform) #waveform?
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size=WINDOW,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
            lambda: tf.size(quantized),
            lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'])

    samples = tf.placeholder(tf.int32)#待初始化的张量占位符
    input_ID = tf.placeholder(tf.int32)
    startime_fastgeration = time.clock()
    if args.fast_generation:
        print ("#########using_fast_generation")
        next_sample = net.predict_proba_incremental(samples, input_ID)

        #print next_sample

    else:
        next_sample = net.predict_proba(samples, input_ID)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)
    endtime_fastgernation = time.clock()
    #print ('fast_generation time {}'.format(endtime_fastgernation - endtime_fastgernation))
    time_of_fast = endtime_fastgernation - startime_fastgeration#1

    start_vari_saver = time.clock()#变量save
    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)
    end_vari_saver = time.clock()
    print ('variables_to_restore{}'.format(end_vari_saver-start_vari_saver))


    starttime_restore = time.clock()#恢复从checkpoint
    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    endtime_restore = time.clock()
    #print ('restore model time{}'.format(endtime_restore - endtime_restore))
    time_of_restore = endtime_restore - starttime_restore#2
    print('%%%%%%%%%%%%{}'.format(time_of_restore))
    #return 0
    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])#namescope(encode)

    quantization_channels = wavenet_params['quantization_channels']
    time_of_seed = 0
    if args.wav_seed:
        start_using_create_seed = time.clock()
        print('#######using_create_seed')
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           quantization_channels)
        waveform = sess.run(seed).tolist() #
        end_using_create_seed = time.clock()
        time_of_seed = end_using_create_seed - start_using_create_seed
        #print ('using create_seed time{}'.format(end_using_create_seed - start_using_create_seed))
    else:
        print('#######not_using_create_seed')
        waveform = np.random.randint(quantization_channels, size=(1,)).tolist()
    predict_of_fast_seed = 0
    if args.fast_generation and args.wav_seed:
        starttime_fast_and_seed = time.clock()
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.
        outputs = [next_sample]
        outputs.extend(net.push_ops) #push_ops是一个列表

        print('Priming generation...')
        for i, x in enumerate(waveform[:-1]):
            if i % 1600 == 0:
                print('Priming sample {}'.format(i), end='\r')
                sys.stdout.flush()
            sess.run(outputs, feed_dict={samples: x, input_ID: args.ID_generation})
        print('Done.')
        endtime_fast_seed = time.clock()
        #print('fast_generation and create_seed time{}'.format(endtime_fast_seed - starttime_fast_and_seed))
        predict_of_fast_seed = predict_of_fast_seed +(endtime_fast_seed - starttime_fast_and_seed)

    #return 0
    last_sample_timestamp = datetime.now()
    predict = 0
    index_begin_generate = 0 if (False == args.fast_generation) else len(waveform)
    startime_total_predict_sample = time.clock()
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
        else:
            if len(waveform) > args.window:#波形大于窗口？
                window = waveform[-args.window:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        starttime_run_net_predict = time.clock()
        prediction = sess.run(outputs, feed_dict={samples: window, input_ID: args.ID_generation})[0]
        endtime_run_net_predict = time.clock()
        print('run net to predict samples per step{}'.format(endtime_run_net_predict - starttime_run_net_predict))
        predict = predict + (endtime_run_net_predict - starttime_run_net_predict)

        # Scale prediction distribution using temperature.尺度化预测!!

        np.seterr(divide='ignore') #某种数据规约尺度化
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after scaling.
        if args.temperature == 1.0:
            np.testing.assert_allclose(prediction, scaled_prediction, atol=1e-5, err_msg='Prediction scaling at temperature=1.0 is not working as intended.')

        sample = np.random.choice(#以概率返回某层通道采样
            np.arange(quantization_channels), p=scaled_prediction)
        waveform.append(sample)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:#以1？？
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            sys.stdout.flush()
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            print('$$$$$$$$$$If we have partial writing, save the result so far')
            out = sess.run(decode, feed_dict={samples: waveform})#有输入要求的tensor
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)
    endtime_total_predicttime = time.clock()
    print('total predic time{}'.format(endtime_total_predicttime - startime_total_predict_sample))
    print('run net predict time{}'.format(predict))

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    '''
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(logdir)
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform[index_begin_generate:], [-1, 1]), input_ID: args.ID_generation})
    writer.add_summary(summary_out)
    '''

    # Save the result as a wav file.
    if args.wav_out_path:
        start_save_wav_time = time.clock()
        out = sess.run(decode, feed_dict={samples: waveform[index_begin_generate:]})
        
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)
        end_save_wave_time = time.clock()
        print('write wave time{}'.format(end_save_wave_time - start_save_wav_time))

    print('time_of_fast_initops{}'.format(time_of_fast))
    print('time_of_restore'.format(time_of_restore))
    print('time_of_fast_and_seed{}'.format(predict_of_fast_seed))
    print('time_of_seed'.format(time_of_seed))
    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
