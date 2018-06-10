import pydicom as dc
import numpy as np
import os
import fnmatch
from random import randint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from xml.etree import ElementTree
from shapely.geometry import Polygon
from shapely.geometry import Point
import util as ut
import platform
import random
from scipy.misc import imresize

# set up the training data file system
class LUNA(object):
    name = 'LUNA'
    colors = 1

    def __init__(self):
        name = platform.node()
        Train_Path = ''
        Eval_Path = ''
        if not name == 'motel':
            Train_Path = './Segmentation_Data'
            Eval_Path = './Segmentation_Data'
        else:
            Train_Path = '/local/scratch/public/sl767/LUNA/Training_Data'
            Eval_Path = '/local/scratch/public/sl767/LUNA/Evaluation_Data'
        # List the existing training data
        self.training_list = ut.find('*.dcm', Train_Path)
        self.training_list_length = len(self.training_list)
        print('Training Data found: ' + str(self.training_list_length))
        self.eval_list = ut.find('*.dcm', Eval_Path)
        self.eval_list_length = len(self.eval_list)
        print('Evaluation Data found: ' + str(self.eval_list_length))
        # List the existing training data according to their xml files
        self.xml_training_list = ut.find('*.xml', Train_Path)
        self.xml_training_list_length = len(self.xml_training_list)
        print('XML Training Data found: ' + str(self.xml_training_list_length))
        self.xml_eval_list = ut.find('*.xml', Eval_Path)
        self.xml_eval_list_length = len(self.xml_eval_list)
        print('XML Evaluation Data found: ' + str(self.xml_eval_list_length))

    def reshape_pic(self, pic):
        pic = ut.normalize_image(pic)
        pic = imresize(pic, [128, 128])
        pic = ut.scale_to_unit_intervall(pic)
        return pic


    # tries hard to get a path to a random legit training set
    def get_random_path(self, training_data = True):
        j = 0
        path = ''
        xml_path = ''
        while j<1000:
            if training_data:
                path = self.training_list[randint(0, self.training_list_length-1)]
            else:
                path = self.eval_list[randint(0, self.eval_list_length-1)]
            k = -1
            while (not path[k] == '/') and k > -1000:
                k = k - 1
            last_number = len(path) + k
            cut_path = path[0:last_number]
            xml_path_list = ut.find('*xml', cut_path)
            if xml_path_list:
                if self.valid_xml(xml_path_list[0]):
                    try:
                        xml_path = xml_path_list[0]
                        dc_file = dc.read_file(path)
                        j = 1000
                    except UnboundLocalError:
                        print('UnboundLocalError caught')
            j = j+1
        if j == 1000:
            print('No legit data found')

        return path, xml_path

    # checks if the xml file is a valid source (has readingSessions instead of CXRreadingSessions)
    def valid_xml(self, xml_path):
        valid = True
        f = ElementTree.parse(xml_path).getroot()
        session = f.findall('{http://www.nih.gov}readingSession')
        if not session:
            valid = False
        return valid


    # mehtode to get a random image with a big nodule
    def load_nodule(self, training_data = True):
        j = 0
        path = ''
        xml_path = ''
        path_list = []
        z_position = 0
        annotations = np.zeros(shape=(512, 512))
        nodules = np.zeros(shape=(512, 512))
        while j < 1000:
            if training_data:
                xml_path = self.xml_training_list[randint(0, self.xml_training_list_length - 1)]
            else:
                xml_path = self.xml_eval_list[randint(0, self.xml_eval_list_length - 1)]
            if self.valid_xml(xml_path):
                f = ElementTree.parse(xml_path).getroot()
                docs = f.findall('{http://www.nih.gov}readingSession')
                nodules = docs[randint(0, len(docs) - 1)].findall('{http://www.nih.gov}unblindedReadNodule')
                nod = nodules[randint(0, len(nodules) - 1)]
                slices = nod.findall('{http://www.nih.gov}roi')
                slice = slices[randint(0, len(slices) - 1)]
                z_position = float(slice[0].text)
                id = slice[1].text
                if len(slice.findall('{http://www.nih.gov}edgeMap')) > 10:
                    # read out annotation map of chosen nodule
                    annotations = np.zeros(shape=(512, 512))
                    nodules = np.zeros(shape=(512, 512))
                    vertices = []
                    for coord in slice.findall('{http://www.nih.gov}edgeMap'):
                        vertices.append((int(coord[0].text), int(coord[1].text)))
                        annotations[int(coord[0].text), int(coord[1].text)] = 1
                        nodules[int(coord[0].text), int(coord[1].text)] = 1
                    try:
                        poly = Polygon(vertices)
                        bnd = poly.bounds
                        for x in range(int(bnd[0]), int(bnd[2] + 1)):
                            for y in range(int(bnd[1]), int(bnd[3] + 1)):
                                point = Point(x, y)
                                if point.within(poly):
                                    nodules[x, y] = 1
                        j = 2000
                    except ValueError:
                        nodules = annotations
                        print('Polygone filling failed. Draw new nodule')
            j = j + 1

        k = -1
        while (not xml_path[k] == '/') and k > -1000:
            k = k - 1
        last_number = len(xml_path) + k
        cut_path = xml_path[0:last_number]
        path_list = ut.find('*dcm', cut_path)

        ### find image in path_list that fits the z position and id of the chosen nodule
        for im_path in path_list:
            dc_file = dc.read_file(im_path)
            image_z = (dc_file[0x0020, 0x0032].value)[2]
            image_id = dc_file[0x0008, 0x0018].value
            if image_z == z_position:
                path = im_path
                assert image_id == id

        dc_file = dc.read_file(path)
        pic = dc_file.pixel_array
        pic = pic - np.amin(pic)
        pic = pic / np.amax(pic)

        return pic, nodules, annotations


    # gets and processes the data
    def get_raw_data(self, path, xml_path):
        # find the slice of the given image
        dc_file = dc.read_file(path)
        z_position = float((dc_file[0x0020, 0x0032].value)[2])
        size = (dc_file.pixel_array).shape

        # read out xml annotation file
        f = ElementTree.parse(xml_path).getroot()
        annotation_list = []
        nodule_list = []
        for child in f.findall('{http://www.nih.gov}readingSession'):
            # the annotation mask of this radiologist
            annotations = np.zeros(shape=size)
            nodules = np.zeros(shape=size)
            # loop over nodules
            for grandchild in child.findall('{http://www.nih.gov}unblindedReadNodule'):
                # loop over 2-dim slices of a single nodule
                for ggc in grandchild.findall('{http://www.nih.gov}roi'):
                    image_z = float(ggc[0].text)
                    # check if current slice has correct z coordinate
                    if image_z == z_position:
                        print('Matching nodule found')
                        vertices = []
                        for coord in ggc.findall('{http://www.nih.gov}edgeMap'):
                            vertices.append((int(coord[0].text), int(coord[1].text)))
                            annotations[int(coord[0].text), int(coord[1].text)] = 1
                        try:
                            poly = Polygon(vertices)
                            bnd = poly.bounds
                            for x in range(int(bnd[0]), int(bnd[2] + 1)):
                                for y in range(int(bnd[1]), int(bnd[3] + 1)):
                                    point = Point(x, y)
                                    if point.within(poly):
                                        nodules[x, y] = 1
                        except ValueError:
                            nodules = annotations
                            print('Polygone filling failed. Using Annotations as binary map.')
            annotation_list.append(annotations)
            nodule_list.append(nodules)

        # renormalize pic
        pic = dc_file.pixel_array
        pic = pic - np.amin(pic)
        pic = pic / np.amax(pic)
        return pic, annotation_list, nodule_list

    # fully processed data
    def load_data(training_data=True):
        pass


    # visualizes the nodule as black and white with red annotations
    def visualize_nodules(self, pic, nod, k):
        size = pic.shape
        three_c = np.zeros(shape=[size[0], size[1], 3])
        for k in range(3):
            three_c[..., k] = pic

        # set red channel to 1 whenever in nodules and set all other channels to 0
        for x in range(size[0]):
            for y in range(size[1]):
                if (nod)[x, y] == 1:
                    three_c[x, y, 0] = 1
                    three_c[x, y, 1] = 0
                    three_c[x, y, 2] = 0

        three_c[..., 0] = three_c[..., 0] + nod
        plt.figure()
        plt.imshow(three_c)
        plt.savefig('Data/Test/' + str(k) + '.jpg')
        plt.close()

