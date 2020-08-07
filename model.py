from collections import namedtuple
from ops import *
from utils import *
from glob import glob
import time
import scipy.io
def build_net_vgg(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias

def build_vgg19(input, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    net = {}
    vgg_rawnet = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = input - np.array([-0.029960784313725397, -0.084086274509804, -0.1847921568627452]).reshape((1, 1, 1, 3))
    net['conv1_1'] = build_net_vgg('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
    net['conv1_2'] = build_net_vgg('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
    net['pool1'] = build_net_vgg('pool', net['conv1_2'])
    net['conv2_1'] = build_net_vgg('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
    net['conv2_2'] = build_net_vgg('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
    net['pool2'] = build_net_vgg('pool', net['conv2_2'])
    net['conv3_1'] = build_net_vgg('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
    net['conv3_2'] = build_net_vgg('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
    net['conv3_3'] = build_net_vgg('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
    net['conv3_4'] = build_net_vgg('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
    net['pool3'] = build_net_vgg('pool', net['conv3_4'])
    net['conv4_1'] = build_net_vgg('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
    net['conv4_2'] = build_net_vgg('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
    net['conv4_3'] = build_net_vgg('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
    net['conv4_4'] = build_net_vgg('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
    net['pool4'] = build_net_vgg('pool', net['conv4_4'])
    net['conv5_1'] = build_net_vgg('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
    net['conv5_2'] = build_net_vgg('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
    net['conv5_3'] = build_net_vgg('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
    net['conv5_4'] = build_net_vgg('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
    net['pool5'] = build_net_vgg('pool', net['conv5_4'])
    return net

def compute_error(real, fake):
    return tf.reduce_mean(tf.abs(real - fake))

def discriminator(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, name='d_h3_conv'), 'd_bn3'))
        h4_logit = conv2d(h3, 1, s=1, name='d_h3_pred')
        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, name='d_h4_conv'), 'd_bn4'))
        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 4, name='d_h5_conv'), 'd_bn5'))
        h6 = lrelu(instance_norm(conv2d(h5, options.df_dim * 2, name='d_h6_conv'), 'd_bn6'))
        h6_logit = conv2d(h6, 1, s=1, name='d_h6_pred')
        return h4_logit,tf.reshape(tf.reduce_mean(h6_logit,axis=[1,2]),[1,1,1,-1])

def generator_resnet(image, options, reuse=False, name="hiding"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim * 4, name='g_r9')
        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def generator_resnet_recon(image, options, reuse=False, name="revealing"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim * 4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim * 4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim * 4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim * 4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim*2, 7, 1, padding='VALID', name='g_pred_c'))

        return pred[:,:,:,:options.output_c_dim],pred[:,:,:,options.output_c_dim:options.output_c_dim*2]


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.patch_size = self.image_size//2
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        self.generator = generator_resnet
        self.generator_recon=generator_resnet_recon

        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.input_c_dim*3],name='collected_images')

        self.alpha = tf.placeholder(tf.float32,[self.batch_size, 1],name='alpha')

        self.real_cover = self.real_data[:, :, :, :self.input_c_dim]
        self.real_message = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim*2]
        self.weighted = self.real_data[:, :, :, self.input_c_dim*2:self.input_c_dim*3]

        self.fake_stegano = self.generator(tf.concat([self.real_cover,self.real_message],axis=3), self.options, False, name="generator_hiding")
        self.recon_cover,self.recon_message = self.generator_recon(self.fake_stegano, self.options, False, name="generator_revealing")
        self.fake_part = tf.random_crop(self.fake_stegano, [1, self.patch_size, self.patch_size, 3])
        self.real_part = tf.random_crop(self.real_cover, [1, self.patch_size, self.patch_size, 3])


        self.Dstegano_fake,self.weighted_est= self.discriminator(self.fake_stegano, self.options, reuse=False, name="discriminator")
        self.Dstegano_fake_part, self.weighted_est_part = self.discriminator(self.fake_part, self.options, reuse=False,name="part")

        self.g_adv=(self.criterionGAN(self.Dstegano_fake, tf.ones_like(self.Dstegano_fake))+abs_criterion(self.weighted_est,tf.zeros_like(self.weighted_est))+
                            self.criterionGAN(self.Dstegano_fake_part, tf.ones_like(self.Dstegano_fake_part))+abs_criterion(self.weighted_est_part,tf.zeros_like(self.weighted_est_part)))

        self.g_est=(abs_criterion(self.weighted_est,tf.zeros_like(self.weighted_est))+abs_criterion(self.weighted_est_part,tf.zeros_like(self.weighted_est_part)))


        with tf.variable_scope("VGG_loss"):
            vgg_real = build_vgg19(self.real_message)
            vgg_fake = build_vgg19(self.recon_message, reuse=True)
            p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2'])
            p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2'])
            p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2'])
            p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2'])
            p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2'])
            self.G_loss = (p1 + p2 + p3 + p4 + p5) * 0.1

        self.g_loss = self.g_adv\
                      + self.G_loss \
                      + self.L1_lambda * abs_criterion(self.real_cover, self.fake_stegano)\
                      + self.L1_lambda * abs_criterion(self.real_cover, self.recon_cover) \
                      + self.L1_lambda * abs_criterion(self.real_message, self.recon_message)

        self.fake_stegano_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.output_c_dim], name='fake_stegano_sample')
        self.fake_stegano_sample_part = tf.random_crop(self.fake_stegano_sample, [1, self.patch_size, self.patch_size, 3])
        self.weighted_part = tf.random_crop(self.weighted, [1, self.patch_size, self.patch_size, 3])

        self.Dcover_real,self.Dcover_est = self.discriminator(self.real_cover, self.options, reuse=True, name="discriminator")
        self.Dstegano_fake_sample,_ = self.discriminator(self.fake_stegano_sample, self.options, reuse=True, name="discriminator")
        _, self.est_real = self.discriminator(self.weighted, self.options, reuse=True,name="discriminator")

        self.d_loss_real = self.criterionGAN(self.Dcover_real, tf.ones_like(self.Dcover_real))
        self.d_loss_fake = self.criterionGAN(self.Dstegano_fake_sample, tf.zeros_like(self.Dstegano_fake_sample))
        self.d_adv_loss = (self.d_loss_real + self.d_loss_fake) / 2
        self.Dcover_est_loss = abs_criterion(self.Dcover_est,tf.zeros_like(self.Dcover_est))
        self.weight_loss = abs_criterion(self.est_real,tf.reshape(self.alpha,[1,1,1,-1]))
        self.d_loss = (self.d_adv_loss + self.Dcover_est_loss+self.weight_loss)
        self.est_loss=self.Dcover_est_loss+self.weight_loss

        self.Dcover_real_part, self.Dcover_est_part = self.discriminator(self.real_part, self.options, reuse=True,name="part")
        self.Dstegano_fake_sample_part, _ = self.discriminator(self.fake_stegano_sample_part, self.options, reuse=True,name="part")
        _, self.est_real_part = self.discriminator(self.weighted_part, self.options, reuse=True,name="part")
        self.d_loss_real_part = self.criterionGAN(self.Dcover_real_part, tf.ones_like(self.Dcover_real_part))
        self.d_loss_fake_part = self.criterionGAN(self.Dstegano_fake_sample_part, tf.zeros_like(self.Dstegano_fake_sample_part))
        self.d_adv_loss_part = (self.d_loss_real_part + self.d_loss_fake_part) / 2
        self.Dcover_est_loss_part = abs_criterion(self.Dcover_est_part, tf.zeros_like(self.Dcover_est_part))
        self.weight_loss_part = abs_criterion(self.est_real_part, tf.reshape(self.alpha, [1, 1, 1, -1]))
        self.d_loss_part = (self.d_adv_loss_part + self.Dcover_est_loss_part + self.weight_loss_part)

        ### G summary
        self.g_adv_sum = tf.summary.scalar("g_adv", self.g_adv)
        self.g_est_sum = tf.summary.scalar("g_est", self.g_est)
        self.G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
        self.g_sum = tf.summary.merge([self.g_adv_sum,self.g_est_sum,self.G_loss_sum])
        ### D summary
        self.d_adv_loss_sum = tf.summary.scalar("d_adv_loss", self.d_adv_loss)
        self.d_adv_loss_part_sum = tf.summary.scalar("d_adv_loss_part", self.d_adv_loss_part)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_part_sum = tf.summary.scalar("d_loss_part", self.d_loss_part)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_real_part_sum = tf.summary.scalar("d_loss_real_part", self.d_loss_real_part)
        self.d_loss_fake_part_sum = tf.summary.scalar("d_loss_fake_part", self.d_loss_fake_part)

        self.d_sum = tf.summary.merge(
            [self.d_adv_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum,
             self.d_adv_loss_part_sum, self.d_loss_real_part_sum, self.d_loss_fake_part_sum,
             self.d_loss_sum,self.d_loss_part_sum]
        )

        self.test_cover = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.input_c_dim], name='test_cover')
        self.test_message = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.output_c_dim], name='test_message')

        self.test_stega = self.generator(tf.concat([self.test_cover,self.test_message],axis=3), self.options, True, name="generator_hiding")
        self.test_cover_recon,self.test_message_recon = self.generator_recon(self.test_stega, self.options, True, name="generator_revealing")

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars=[var for var in t_vars if 'discriminator' in var.name]
        self.part_vars = [var for var in t_vars if 'part' in var.name]

        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        self.part_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss_part, var_list=self.part_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir,"logs"), self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            data_cover = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train_cover'))
            data_message = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/train_message'))
            np.random.shuffle(data_cover)
            np.random.shuffle(data_message)
            batch_idxs = min(min(len(data_cover), len(data_message)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(data_cover[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       data_message[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images,alpha = load_train_data(batch_files[0], args.load_size, args.fine_size)
                batch_images = [batch_images]
                alpha = [alpha]
                alpha = np.reshape(alpha,[self.batch_size,1])
                batch_images = np.array(batch_images).astype(np.float32)
                # Update G network and record fake outputs
                fake_stegano,_,g_loss,g_est,g_adv,G_loss,summary_str = self.sess.run(
                    [self.fake_stegano, self.g_optim,self.g_loss,self.g_est,self.g_adv,self.G_loss, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr,self.alpha:alpha})
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _,_,d_loss,est_loss,d_summary_str = self.sess.run(
                    [self.d_optim,self.part_optim, self.d_loss,self.est_loss,self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_stegano_sample: fake_stegano,
                               self.lr: lr,self.alpha:alpha})
                self.writer.add_summary(d_summary_str, counter)
                counter += 1
                print(("Epoch:[%2d][%4d/%4d] time: %4.4f  g_loss: %4.4f  g_est: %4.4f g_adv: %4.4f G_loss: %4.4f d_loss: %4.4f est_loss: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time,g_loss,g_est,g_adv,G_loss,d_loss,est_loss)))
                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "stegano.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        data_cover = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test_cover'))
        data_message = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test_message'))
        np.random.shuffle(data_cover)
        np.random.shuffle(data_message)
        batch_files = list(zip(data_cover[:self.batch_size], data_message[:self.batch_size]))
        sample_images,_ =load_train_data(batch_files[0],is_testing=True)
        sample_images=[sample_images]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_stegano,rec_cover,rec_message = self.sess.run(
            [self.fake_stegano,self.recon_cover,self.recon_message],
            feed_dict={self.real_data: sample_images}
        )
        real_cover = sample_images[:, :, :, :self.input_c_dim]
        real_message = sample_images[:, :, :, self.input_c_dim:self.input_c_dim*2]
        weighted = sample_images[:, :, :, self.input_c_dim*2:self.input_c_dim*3]

        merge = np.concatenate([real_cover,real_message,weighted, fake_stegano,rec_cover,rec_message], axis=2)
        check_folder('./{}/{:02d}'.format(sample_dir, epoch))
        save_images(merge, [self.batch_size, 1],
                    './{}/{:02d}/{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        data_cover = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test_cover'))
        data_message = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test_message'))
        for sample_file in data_message:
            print('Processing image: ' + sample_file)
            random_idx=np.random.randint(0,len(data_cover))
            cover_path=data_cover[random_idx]
            sample_image = [load_test_data(cover_path,sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            jpg_name=os.path.basename(sample_file)
            image_path = os.path.join(args.test_dir,'{0}_{1}'.format("merge",jpg_name[:-4]+".png" ))
            test_A,test_B,fake_img,recon_cover,recon_message = self.sess.run([self.test_cover,self.test_message,self.test_stega,self.test_cover_recon,self.test_message_recon],
                                                               feed_dict={self.test_cover: sample_image[:,:,:,:3],self.test_message:sample_image[:,:,:,3:6]})
            merge=np.concatenate([test_A,test_B,fake_img,recon_cover,recon_message],axis=2)
            save_images(merge, [1, 1], image_path)
            split_path = os.path.join(args.test_dir, jpg_name[:-4] + ".png")
            save_images(fake_img, [1, 1], split_path)

    def test_reverse(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        data_stegano = glob('{}/*.*'.format(args.stegano_dir))
        for sample_file in data_stegano:
            print('Processing image: '+sample_file)
            sample_image = [load_reverse_data( sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            jpg_name = os.path.basename(sample_file)
            image_path = os.path.join(args.recon_dir, '{}'.format(jpg_name[:-4] + ".png"))
            recon_cover,recon_message = self.sess.run(
                [self.test_cover_recon,self.test_message_recon],
                feed_dict={self.test_stega: sample_image})
            merge = np.concatenate([recon_cover,recon_message], axis=2)
            save_images(merge, [1, 1], image_path)