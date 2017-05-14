import tensorflow as tf
import time
from .ops import causal_conv, mu_law_encode

#count_generator_con =0
#_generator_dilation_layer_count =0
def create_variable(name, shape):#创建一个卷积过滤器用特定的过滤器和名字
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)#tf.Variable可以保存变量的名字和形状
    print('---------------------------variable---------------------')
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    print('--------------------------bias------------------------')
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch, input_IDs)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 ID_channels=109,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            ID_channels: How many speakers to use for audio
                training and the corresponding one-hot encoding.
                Default: 109 (VCTK).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.ID_channels = ID_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width

        self.variables = self._create_variables() #!!

    def _create_variables(self):#这里涉及到的变量看懂
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''
        start = time.clock()
        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.quantization_channels
                    initial_filter_width = self.filter_width #"filter_width":2
                layer['filter'] = create_variable(#filter:(2,256,32)
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer #var是一个多层字典

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])
                        current['filter_ID_bias'] = create_bias_variable(
                            'filter_ID_bias',
                            [self.ID_channels,
                             self.dilation_channels])
                        current['gate_ID_bias'] = create_bias_variable(
                            'gate_ID_bias',
                            [self.ID_channels,
                             self.dilation_channels])
                        if self.use_biases:
                            '''
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])'''
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current
        end = time.clock()
        print('#######creat variables time{}'.format(end - start))
        return var

    def _create_causal_layer(self, input_batch):#网络结构图看懂？？
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        start = time.clock()
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)
        end =time.clock()
        print('###########create causal layer{}'.format(end - start))

    def _create_dilation_layer(self, input_batch, input_IDs, layer_index, dilation):#扩展层
        '''Creates a single causal dilated convolution layer.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output.
        '''
        start = time.clock()
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        filter_ID_bias = variables['filter_ID_bias']
        gate_ID_bias = variables['gate_ID_bias']
        #input_IDs，gate_ID_bias为什么会用在乘法计算中
        filter_bias = tf.matmul(input_IDs, filter_ID_bias)
        gate_bias = tf.matmul(input_IDs, gate_ID_bias)

        # to be compatible with batchsize > 1
        shape = tf.shape(filter_bias)
        filter_bias = tf.reshape(filter_bias, [shape[0], 1, shape[1]])
        shape = tf.shape(gate_bias)
        gate_bias = tf.reshape(gate_bias, [shape[0], 1, shape[1]])

        conv_filter = tf.add(conv_filter, filter_bias)
        conv_gate = tf.add(conv_gate, gate_bias)

        '''
        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)'''

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate) #这就是z那个公式

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(#为什么要将输出的out和weight_skip进行一个卷积？看论文
            out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        layer = 'layer{}'.format(layer_index)
        tf.histogram_summary(layer + '_filter', weights_filter)
        tf.histogram_summary(layer + '_gate', weights_gate)
        tf.histogram_summary(layer + '_dense', weights_dense)
        tf.histogram_summary(layer + '_skip', weights_skip)
        if self.use_biases:
            tf.histogram_summary(layer + '_biases_filter', filter_bias)
            tf.histogram_summary(layer + '_biases_gate', gate_bias)
            tf.histogram_summary(layer + '_biases_dense', dense_bias)
            tf.histogram_summary(layer + '_biases_skip', skip_bias)

        end = time.clock()
        print('#####create dilation layer{}'.format(end - start))
        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        start = time.clock()
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights) #看懂这里的意思
        end = time.clock()
        print('#############_generator_conv{}'.format(end - start))
        #count_generator_con = count_generator_con + 1
        #print count_generator_con
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        start = time.clock()
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        end = time.clock()
        print('############_generator_causal_layer{}'.format(end - start))
        return output

    def _generator_dilation_layer(self, input_batch, input_IDs, state_batch, layer_index,
                                  dilation):
        start = time.clock()
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        filter_ID_bias = variables['filter_ID_bias']
        gate_ID_bias = variables['gate_ID_bias']

        filter_bias = tf.matmul(input_IDs, filter_ID_bias)
        gate_bias = tf.matmul(input_IDs, gate_ID_bias)

        # to be compatible with batchsize > 1 ? only support batchsize = 1 in generation
        #shape = tf.shape(filter_bias)
        #filter_bias = tf.reshape(filter_bias, [shape[0], 1, shape[1]])
        #shape = tf.shape(gate_bias)
        #gate_bias = tf.reshape(gate_bias, [shape[0], 1, shape[1]])
        
        output_filter = output_filter + filter_bias
        output_gate = output_gate + gate_bias

        '''
        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']
        '''

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']
        end = time.clock()
        print('################_generator_dilation_layer{}'.format(end - start))
       # _generator_dilation_layer_count = _generator_dilation_layer_count +1
        #print _generator_dilation_layer_count
        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, input_IDs):#
        '''Construct the WaveNet network.'''
        start = time.clock()
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution::普通卷积
        if self.scalar_input:
            initial_channels = 1
        else:
            initial_channels = self.quantization_channels #256

        current_layer = self._create_causal_layer(current_layer)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):#dilation_stack
            for layer_index, dilation in enumerate(self.dilations):#每一层的dilation
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, input_IDs, layer_index, dilation)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            tf.histogram_summary('postprocess1_weights', w1)#w1的图表
            tf.histogram_summary('postprocess2_weights', w2)
            if self.use_biases:
                tf.histogram_summary('postprocess1_biases', b1)
                tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)#每一层的跳跃连接输出累加
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)
        end = time.clock()
        print('#############_create_network{}'.format(end - start))
        return conv2

    def _create_generator(self, input_batch, input_IDs):
        '''Construct an efficient incremental generator.'''
        start = time.clock()
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batch_size, self.quantization_channels))
        init = q.enqueue_many(
            tf.zeros((1, self.batch_size, self.quantization_channels)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
                            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):

                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, input_IDs, current_state, layer_index, dilation)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2
        end = time.clock()
        print('######################_create_generator{}'.format(end - start))
        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        start = time.clock()
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        end = time.clock()
        print('#########one_hot{}'.format(end -start))
        return encoded

    def _one_hot_ID(self, input_IDs):#
        '''One-hot encodes the IDs.
        '''
        start = time.clock()
        with tf.name_scope('one_hot_ID_encode'):
            encoded = tf.one_hot(
                input_IDs,
                depth=self.ID_channels,#ID_channels即ID的通道数
                dtype=tf.float32)
            shape = [self.batch_size, self.ID_channels]#转成这种维度
            encoded = tf.reshape(encoded, shape)
        end = time.clock()
        print('########one_hot_ID{}'.format(end - start))
        return encoded

    def predict_proba(self, waveform, input_IDs, name='wavenet'):#计算下一个采样的概率分布
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        start = time.clock()
        with tf.name_scope(name):
            if self.scalar_input:#标量输入
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])#转成二维形式
            else:
                encoded = self._one_hot(waveform)
            encoded_input_IDs = self._one_hot_ID(input_IDs)
            raw_output = self._create_network(encoded, encoded_input_IDs)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])#调整为256维
            # Cast to float64 to avoid bug in TensorFlow
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            end = time.clock()
            print('##########predict_proba{}'.format(end - start))
            return tf.reshape(last, [-1])

    def predict_proba_incremental(self, waveform, input_IDs, name='wavenet'):#网络的输出feed回输入的更快输出
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        start = time.clock()
        if self.filter_width > 2:#过滤器宽度最多为2,为什么？
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            #维度变成waveform.demension*256
            encoded = tf.one_hot(waveform, self.quantization_channels)
            encoded = tf.reshape(encoded, [-1, self.quantization_channels])
            encoded_input_IDs = self._one_hot_ID(input_IDs)
            raw_output = self._create_generator(encoded, encoded_input_IDs)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            end = time.clock()
            print('#########predict_proba_incremental{}'.format(end - start))
            return tf.reshape(last, [-1])
        
    def count_viewfield_size(self):
        '''Computes the view field size'''
        max_dilations = max(self.dilations)#512
        stacknum = self.dilations.count(max_dilations)
        viewfield_size = (max_dilations * 2 - 1) * stacknum + 1 #为什么是这样的
        viewfield_size += 1  # add one for the first causal layer
        return viewfield_size

    def loss(self, #返回的是自编码的损失
             input_batch,
             input_IDs,
             l2_regularization_strength=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        start = time.clock()
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            input_batch = mu_law_encode(input_batch,
                                        self.quantization_channels)

            encoded = self._one_hot(input_batch)
            if self.scalar_input:#标量输入的形式
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded
            encoded_input_IDs = self._one_hot_ID(input_IDs)#input_IDs的表示形式

            raw_output = self._create_network(network_input, encoded_input_IDs)

            with tf.name_scope('loss'):
                # Shift original input left by one sample, which means that
                # each output sample has to predict the next input sample.
                # Cut both raw input and output 
                viewfield_size = self.count_viewfield_size()
                cuted_input = tf.slice(encoded, [0, viewfield_size, 0],
                                   [-1, tf.shape(encoded)[1] - viewfield_size, -1])
                cuted_output = tf.slice(raw_output, [0, viewfield_size - 1, 0],
                                   [-1, tf.shape(raw_output)[1] - viewfield_size, -1])
                prediction = tf.reshape(cuted_output, [-1, self.quantization_channels])
                label = tf.reshape(cuted_input, [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits(prediction, label)
                l1 = tf.abs( tf.sub(tf.argmax(prediction,1), tf.argmax(label,1)) ) 
                reduced_loss = tf.reduce_mean(loss)
                reduced_l1 = tf.reduce_mean(tf.cast(l1, tf.float32))

                tf.scalar_summary('loss', reduced_loss)
                tf.scalar_summary('l1', reduced_l1)

                if l2_regularization_strength is None:
                    end = time.clock()
                    print('########loss{}'.format(end - start))
                    return reduced_loss, reduced_l1
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.scalar_summary('l2_loss', l2_loss)
                    tf.scalar_summary('total_loss', total_loss)
                    end = time.clock()
                    print('########loss{}'.format(end - start))

                    return total_loss, reduced_l1
