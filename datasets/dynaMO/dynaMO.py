# this is supposed to be a dynamic MNIST creator
# dynaMO is short for dynamic MNIST with Occlusion
# it creates stereoscopic views of two or three MNIST digits
# that can be modified by simply changing the viewpoint
# Parallax

import torch
from torchvision import datasets, transforms
import numpy as np
import scipy.ndimage
from skimage import io
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import time
import threading
import errno
import os
from get_mnist import get



# commandline arguments
# -----

parser = argparse.ArgumentParser()
parser.add_argument(
     "-npr",
     "--nproliferation",
     type=int,
     default=10,
     help='number of images per image')
parser.add_argument(
     "-nth",
     "--nthreads",
     type=int,
     default=2,
     help='number of threads for the builder')
parser.add_argument(
     "-tst",
     "--timesteps",
     type=int,
     default=15,
     help='how many timesteps are visualized')
parser.add_argument(
     "-oformat",
     "--outputformat",
     type=str,
     default='bmp',
     help='output format (.bmp, .png, .jpg, lmdb format)')
parser.add_argument(
     "-n",
     "--name",
     type=str,
     default='dynamo',
     help='name of the dataset, export folder')
parser.add_argument('--classduplicates', dest='classduplicates', action='store_true')
parser.add_argument('--no-classduplicates', dest='classduplicates', action='store_false')
parser.set_defaults(classduplicates=True)

parser.add_argument('--testrun', dest='testrun', action='store_true')
parser.add_argument('--no-testrun', dest='testrun', action='store_false')
parser.set_defaults(testrun=False)

parser.add_argument('--zoom', dest='zoom', action='store_true')
parser.add_argument('--no-zoom', dest='zoom', action='store_false')
parser.set_defaults(zoom=True)

parser.add_argument('--strips', dest='strips', action='store_true')
parser.add_argument('--no-strips', dest='strips', action='store_false')
parser.set_defaults(strips=True)

parser.add_argument('--interactive', dest='interactive', action='store_true')
parser.add_argument('--no-interactive', dest='interactive', action='store_false')
parser.set_defaults(interactive=False)
args = parser.parse_args()


def mkdir_p(path):
    """
    mkdir_p takes a string path and creates a directory at this path if it
    does not already exist.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def getch():
    import termios
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()

class dynaMOSample(object):
    """docstring for dynaMOSample."""

    def __init__(self, targets, labels, cam_pos, xyz_tars, movement=None):
        super(dynaMOSample, self).__init__()
        self.targets = targets
        self.labels = labels
        self.cam_pos = cam_pos
        self.xyz_tars = xyz_tars
        self.state = self.generate_state(self.targets, self.xyz_tars)
        self.movement = movement if movement else self.generate_movement(timesteps=10, stepsize=0.002, type='random')

    def generate_movement(self, timesteps, stepsize, type):
        movement_x = [0]*timesteps
        movement_y = [0]*timesteps
        if 'l' in type:
            movement_x = [-stepsize]*timesteps
        if 'r' in type:
            movement_x = [stepsize]*timesteps
        if 'u' in type:
            movement_y = [-stepsize]*timesteps
        if 'd' in type:
            movement_y = [stepsize]*timesteps
        if type == 'random':
            movement_x = [stepsize]*np.random.choice([0,1,-1], timesteps)
            movement_y = [stepsize]*np.random.choice([0,1,-1], timesteps)
        movement = [movement_x, movement_y]
        self.movement = movement
        return movement

    def generate_sequence_state(self):
        seq_occlusion_percentage = []
        seq_occlusion_percentage.append(self.occlusion_percentage)
        
        seq_state = np.zeros([self.state.shape[0], self.state.shape[1]*(len(self.movement[0])+1)], dtype=np.uint8)
        seq_state[:,:self.state.shape[0]] = self.state
        
        for i in range(len(self.movement[0])):
            _,_,_ = self.move_camera(self.movement[0][i], self.movement[1][i])
            seq_occlusion_percentage.append(self.occlusion_percentage)
            seq_state[:,(i+1)*self.state.shape[0]:(i+2)*self.state.shape[0]] = self.state
        self.sequence_state = seq_state
        self.sequence_occlusion_percentage = seq_occlusion_percentage
        pass
    
    def save_sequence_state_to_image(self, filename, format):
        "Generate a single image from a sequence state"
        io.imsave('{}.{}'.format(filename, format), self.sequence_state)
        pass
    
    def save_sequence_state_to_images(self, filename, format):
        "Generate multiple images from a sequence state"
        occlusions = [str(int(np.round(o,2)*100)) for o in self.sequence_occlusion_percentage] 
        filename_new = filename.rsplit('_',3)[0] + '_0_' + filename.rsplit('_',3)[1] + \
            '_' + occlusions[0] + '_' + filename.rsplit('_',3)[2] + \
            '_' + filename.rsplit('_',3)[3]
        state = self.sequence_state[:,:self.state.shape[0]]
        io.imsave('{}.{}'.format(filename_new, format), state)
        for i in range(len(self.movement[0])):
            filename_new =  filename.rsplit('_',3)[0] + '_{}_'.format(i+1) + filename.rsplit('_',3)[1] + \
            '_' + occlusions[i+1] + '_' + filename.rsplit('_',3)[2] + \
            '_' + filename.rsplit('_',3)[3]
            state = self.sequence_state[:,(i+1)*self.state.shape[0]:(i+2)*self.state.shape[0]]
            io.imsave('{}.{}'.format(filename_new,  format), state)
        pass

    def get_zoom(self, z_tar, true_size = .6):
        canvas_size = 1 * 2 * z_tar # tan(45) = 1
        return true_size/canvas_size

    def convert_cm_to_px(self, x_or_y, z_tar, tar_pixel_size):
        canvas_size = 1 * 2 * z_tar
        ret = x_or_y/canvas_size * 32 + 48 - tar_pixel_size/2
        return int(round(ret, 0))

    def generate_state(self, targets, xyz_tars):
        canvas = np.zeros([3,128,128], dtype=np.uint8)
        for t in range(targets.shape[0]):
            if args.zoom:
                tar = scipy.ndimage.zoom(targets[t], self.get_zoom(xyz_tars[t][-1]))
            else:
                tar = targets[t]
            x_position = self.convert_cm_to_px(xyz_tars[t][1],xyz_tars[t][-1], tar.shape[0])
            y_position = self.convert_cm_to_px(xyz_tars[t][0],xyz_tars[t][-1], tar.shape[0])
            try:
                canvas[-1-t, x_position:x_position+tar.shape[0], y_position:y_position+tar.shape[1]] = tar
            except(ValueError):
                print("[INFO]: End of canvas.")
                return self.state
        target_pixels = canvas[0, 48:48+32, 48:48+32][canvas[0, 48:48+32, 48:48+32] != 0].shape[0]
        # Make sure there is a notion of occlusion and order
        canvas_o = np.copy(canvas)
        for t in range(targets.shape[0]-1):
            canvas_o[t, :, :][np.max(
                canvas[t+1:, :, :], axis=0, keepdims=False) != 0] = 0
        merged = np.max(canvas_o, 0, keepdims=False)
        output = merged[48:48+32,48:48+32]

        try:
            self.occlusion_percentage = \
                (target_pixels -
                    canvas_o[0, 48:48+32, 48:48+32]
                    [canvas_o[0, 48:48+32, 48:48+32] != 0]
                    .shape[0])/target_pixels
        except(ZeroDivisionError):
            print("[INFO]: Division by Zero, Occlusion might not make sense")
            self.occlusion_percentage = 1.0
        #self.occlusion_percentage = 0 if (self.occlusion_percentage < 0) else self.occlusion_percentage
        #print(target_pixels, canvas_o[0, 48:48+32, 48:48+32]
        #[canvas_o[0, 48:48+32, 48:48+32] != 0].shape[0])
        return output

    def move_camera(self, x_move, y_move):
        self.cam_pos[1] += x_move
        self.cam_pos[0] += y_move
        for t in range(len(self.xyz_tars)):
            self.xyz_tars[t][0] -= x_move / self.xyz_tars[t][-1] * 2
            self.xyz_tars[t][1] -= y_move / self.xyz_tars[t][-1] * 2
        self.state = self.generate_state(self.targets, self.xyz_tars)
        return self.cam_pos[0], self.cam_pos[1], self.xyz_tars

    def show_image(self):
        plt.imshow(self.state, cmap='Greys_r')
        plt.show()

    def animation_init_function(self):
        self.ax = plt.axes(title='target={}, occ.={}'.format(self.labels[-1], np.round(self.occlusion_percentage,2)))
        self.im = self.ax.imshow(self.state, cmap="Greys_r")
        self.im.set_data(self.state)
        return [self.im]

    def animation_function(self, i):
        _,_,_ = self.move_camera(self.movement[0][i], self.movement[1][i])
        self.im.set_array(self.state)
        self.ax.set_title('target={}, occ.={}'.format(self.labels[-1], np.round(self.occlusion_percentage,2)))
        return [self.im]


    def animate_parallax(self, output_name, frames, interval):
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, self.animation_function, init_func=self.animation_init_function,
                                       frames=frames, interval=interval, blit=True)
        anim.save('{}.mp4'.format(output_name), fps=30, extra_args=['-vcodec', 'libx264'])
        #plt.show()
        print('done.')


class dynaMOBuilder(object):
    """docstring for dynaMOBuilder."""

    def __init__(self, num_class=10, class_duplicates=True, timesteps=15, n_proliferation=10, n_threads=2):
        super(dynaMOBuilder, self).__init__()
        self.num_class = num_class
        self.n_proliferation = n_proliferation
        self.n_threads = n_threads
        self.class_set = set(range(self.num_class))
        self.class_duplicates = class_duplicates
        self.timesteps = timesteps

    def build(self, target='train', dpath='./dynamo/', output_format='bmp'):
        if target=='train':
            data, labels, _, _  = get()
        else:
            _, _, data, labels  = get()

        self.target = target
        self.dpath = dpath
        self.lock = threading.Lock()
            
        def work(data, labels, thread_number, starting_idx):
            print("Task {} assigned to thread: {}".format(thread_number, threading.current_thread().name))
            print("ID of process running task {}: {}".format(thread_number, os.getpid()))
            for i, mnist_image in enumerate(data):
                if not self.class_duplicates:
                    ks = []
                    cands_of_same_class = np.where(labels==labels[i])
                    choiceset = set(range(data.shape[0])).difference(set(cands_of_same_class[0]))
                    js = np.random.choice(np.array(list(choiceset)), self.n_proliferation)

                    for j in js:
                        cands_of_same_class = np.where(labels==labels[j])
                        subchoiceset = choiceset.difference(set(cands_of_same_class[0]))
                        ks.append(np.random.choice(np.array(list(subchoiceset))))
                    ks = np.array(ks)
                else:
                    js = np.random.choice(data.shape[0], self.n_proliferation)
                    ks = np.random.choice(data.shape[0], self.n_proliferation)

                for p in range(self.n_proliferation):
                    choice = [ks[p], js[p], i]
                    tars = data[choice].reshape([-1, 28, 28])
                    labs = labels[choice]
                    cam_x_pos, cam_y_pos = 0, 0
                    xyz_tars = [[np.random.uniform()*0.3+0.15,np.random.uniform()*0.3+0.15,.3],
                                [np.random.uniform()*0.4+0.2,np.random.uniform()*0.4+0.2,.4],
                                [np.random.uniform()*0.5+0.25,np.random.uniform()*0.5+0.25,.5]]
                    sample = dynaMOSample(tars, labs, [cam_x_pos, cam_y_pos], xyz_tars)
                    typechoice = np.random.choice(['u', 'd', 'l', 'r', 'ur', 'ul', 'dr', 'dl'])
                    _ = sample.generate_movement(self.timesteps, 0.002, typechoice)
                    filename = self.dpath + '/{}/{}/{}_{}_{}{}{}_{}_{}'.format(self.target, sample.labels[-1], starting_idx + i, p, sample.labels[0], sample.labels[1], sample.labels[2], thread_number, p + i*self.n_proliferation)
                    mkdir_p(filename.rsplit('/', 1)[0])
                    sample.generate_sequence_state()
                    if args.strips:
                        sample.save_sequence_state_to_image(filename, output_format)
                    # elif args.grids:
                    #     gridlist.append(sample.sequence_state)
                    else:
                        sample.save_sequence_state_to_images(filename, output_format)
                    #print(" " * 80 + "\r" +
                    #    '[INFO]: Class {}: ({} / {}) \t Total: ({} / {})'.format(sample.labels[-1], p+1, self.n_proliferation, p+1 + i*self.n_proliferation, data.shape[0]*self.n_proliferation),  end="\r")
                    if (p+1 + i*self.n_proliferation)%100 == 0:
                        print("[THREAD {}]: ({} / {}) images done".format(threading.current_thread().name, p+1 + i*self.n_proliferation, data.shape[0]*self.n_proliferation))
                    # 
                    # if args.grids:
                    #     # stack gridlist
                    #     # save gridlist to file
            return None

        if args.testrun:
            data=data[:30]
            labels = labels[:30]
        # split the data
        datasize_per_thread = data.shape[0]//self.n_threads
        # establish threads
        threadlist = []
        for t in range(self.n_threads - 1):
            threadlist.append(threading.Thread(target=work, args=(data[t*datasize_per_thread:(t+1)*datasize_per_thread], labels[t*datasize_per_thread:(t+1)*datasize_per_thread], t, t*datasize_per_thread), name='t{}'.format(t)))
        threadlist.append(threading.Thread(target=work, args=(data[(self.n_threads - 1)*datasize_per_thread:], labels[(self.n_threads - 1)*datasize_per_thread:], (self.n_threads - 1), (self.n_threads - 1)*datasize_per_thread), name='t{}'.format((self.n_threads - 1))))

        for t in range(self.n_threads):
            threadlist[t].start()
        for t in range(self.n_threads):
            threadlist[t].join()
    
    def build_mnist_only(self, target='train', dpath='./mnist/', output_format='bmp'):
        if target=='train':
            data, labels, _, _  = get()
        else:
            _, _, data, labels  = get()

        self.target = target
        self.dpath = dpath
            
        def work(data, labels, thread_number):
            print("Task {} assigned to thread: {}".format(thread_number, threading.current_thread().name))
            print("ID of process running task {}: {}".format(thread_number, os.getpid()))

            for i, mnist_image in enumerate(data):
                filename = self.dpath + '/{}/{}/{}_{}i{}'.format(self.target, labels[i],labels[i], thread_number, i)
                mkdir_p(filename.rsplit('/', 1)[0])
                io.imsave('{}.{}'.format(filename, output_format), mnist_image.reshape([28,28]))
                if (i)%100 == 0:
                    print("[THREAD {}]: ({} / {}) images done".format(threading.current_thread().name, i, data.shape[0]))
            return None
            
        if args.testrun:
            data=data[:30]
            labels = labels[:30]
        # split the data
        datasize_per_thread = data.shape[0]//self.n_threads
        # establish threads
        threadlist = []
        for t in range(self.n_threads - 1):
            threadlist.append(threading.Thread(target=work, args=(data[t*datasize_per_thread:(t+1)*datasize_per_thread], labels[t*datasize_per_thread:(t+1)*datasize_per_thread], t, t*datasize_per_thread), name='t{}'.format(t)))
        threadlist.append(threading.Thread(target=work, args=(data[(self.n_threads - 1)*datasize_per_thread:], labels[(self.n_threads - 1)*datasize_per_thread:], (self.n_threads - 1), (self.n_threads - 1)*datasize_per_thread), name='t{}'.format((self.n_threads - 1))))

        for t in range(self.n_threads):
            threadlist[t].start()
        for t in range(self.n_threads):
            threadlist[t].join()
    
    def generate_metadata(self, dpath):       
        import pandas as pd
        df = pd.DataFrame(columns=['relative_file_path', 'digit_id', 'contrast', 'time', 'labels', 'occlusion_percentage', 'thread', 'iteration'])
        classlist = os.listdir(dpath)
        try:
            classlist.remove('.DS_Store')  # macOS Finder metadata
        except(ValueError):
            pass
        for classname in classlist:
            filelist = os.listdir(os.path.join(dpath, classname))
            try:
                filelist.remove('.DS_Store')
            except(ValueError):
                pass
            for filename in filelist:
                list_of_entries = ['{}/'.format(classname) + filename] + filename.split('.')[0].split('_')
                
                df = df.append(
                    {'relative_file_path':list_of_entries[0],
                    'digit_id':list_of_entries[1],
                    'contrast':list_of_entries[2],
                    'time':list_of_entries[3],
                    'labels':list_of_entries[4],
                    'occlusion_percentage':list_of_entries[5],
                    'thread':list_of_entries[6],
                    'iteration':list_of_entries[7]}, ignore_index=True)                 
        
        df.to_csv(os.path.join(dpath, 'metadata.zip'), index=False)
        pass

    def generate_metadata_old(self, dpath):
        import csv
        import zipfile        
        with open(os.path.join(dpath, 'metadata.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['relative_file_path', 'digit_id', 'contrast', 'time', 'labels', 'occlusion_percentage', 'thread', 'iteration'])
            classlist = os.listdir(dpath)
            classlist.remove('metadata.csv')
            try:
                classlist.remove('.DS_Store')  # macOS Finder metadata
            except(ValueError):
                pass
            for classname in classlist:
                filelist = os.listdir(os.path.join(dpath, classname))
                try:
                    filelist.remove('.DS_Store')
                except(ValueError):
                    pass
                for filename in filelist:
                    writer.writerow(['{}/'.format(classname) + filename] + filename.split('.')[0].split('_'))
        with zipfile.ZipFile(os.path.join(dpath, 'metadata.zip'), mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
            zip.write(os.path.join(dpath, 'metadata.csv'))
            zip.close()
        os.remove(os.path.join(dpath, 'metadata.csv'))
        pass


def create_sample_animations(N=10):
    train, train_labels, test, test_labels = get()
    for i in range(N):
        choice = np.random.choice(train.shape[0], 3)
        targets = train[choice].reshape([-1, 28, 28])
        labels = train_labels[choice]
        cam_x_pos, cam_y_pos = 0, 0
        xyz_tars = [[np.random.uniform()*0.3+0.15,np.random.uniform()*0.3+0.15,.3],
                    [np.random.uniform()*0.4+0.2,np.random.uniform()*0.4+0.2,.4],
                    [np.random.uniform()*0.5+0.25,np.random.uniform()*0.5+0.25,.5]]
        move = 0.002
        movement_x = [move]*10 + [-move]*20 + [move]*10 + [0]*40 + [move]*10 + [-move]*20 +[move]*10
        movement_y = [0]*40 + [move]*10 + [-move]*20 + [move]*10 + [move]*10 + [-move]*20 +[move]*10
        movement = [movement_x, movement_y]
        sample = dynaMOSample(targets, labels, [cam_x_pos, cam_y_pos], xyz_tars, movement)
        sample.animate_parallax('parallax_{}'.format(i), frames=len(sample.movement[0]), interval=50)

if __name__ == "__main__":

    if args.interactive:
        train, train_labels, test, test_labels = get()

        def resample():
            choice = np.random.choice(train.shape[0], 3)
            targets = train[choice].reshape([-1, 28, 28])
            labels = train_labels[choice]
            cam_x_pos, cam_y_pos = 0, 0
            xyz_tars = [[np.random.uniform()*0.3+0.15,np.random.uniform()*0.3+0.15,.3],
                        [np.random.uniform()*0.4+0.2,np.random.uniform()*0.4+0.2,.4],
                        [np.random.uniform()*0.5+0.25,np.random.uniform()*0.5+0.25,.5]]


            return dynaMOSample(targets, labels, [cam_x_pos, cam_y_pos], xyz_tars)

        sample = resample()
        switch = True
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, title='target={}, occ.={}'.format(sample.labels[-1], np.round(sample.occlusion_percentage,2)))
        image = ax.imshow(sample.state, cmap='Greys_r')
        print("navigate the camera using 'W', 'A', 'S', 'D', press 'Q' to quit.")
        while switch:
            key = getch()
            #print(key)
            movement = 0.01
            if key == 'w':
                sample.move_camera(0, -movement)
            if key == 's':
                sample.move_camera(0, +movement)
            if key == 'a':
                sample.move_camera(-movement, 0)
            if key == 'd':
                sample.move_camera(+movement, 0)
            if key == 'q':
                switch = False
            if key == 'r':
                sample = resample()
            ax.set_title('target={}, occ.={}'.format(sample.labels[-1], np.round(sample.occlusion_percentage,2)))
            image.set_data(sample.state)
            fig.canvas.draw()
            plt.pause(.01)

    else:
        data_path = './data/'
        # data_path = '/home/aecgroup/aecdata/contrastive_learning/'
        
        b = dynaMOBuilder(class_duplicates=args.classduplicates, timesteps=args.timesteps, n_proliferation=args.nproliferation, n_threads=args.nthreads)
        b.build(target='train', output_format=args.outputformat,
            dpath=data_path + '{}'.format(args.name))
        b.generate_metadata(dpath=data_path + '{}/train/'.format(args.name))
        b.build(target='test', output_format=args.outputformat,
            dpath=data_path + '{}'.format(args.name))
        b.generate_metadata(dpath=data_path + '{}/test/'.format(args.name))
