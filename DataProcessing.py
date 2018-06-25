import pydicom as dc
import numpy as np
import os
import fnmatch
from random import randint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
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
        self.source_path = ''
        if not name == 'motel':
            Train_Path = './Segmentation_Data'
            Eval_Path = './Segmentation_Data'
            self.source_path = './Segmentation_Data'
        else:
            Train_Path = '/local/scratch/public/sl767/LUNA/Training_Data'
            Eval_Path = '/local/scratch/public/sl767/LUNA/Evaluation_Data'
            self.source_path = '/local/scratch/public/sl767/LUNA/'
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


    # checks if the xml file is a valid source (has readingSessions instead of CXRreadingSessions)
    def valid_xml(self, xml_path):
        valid = True
        f = ET.parse(xml_path).getroot()
        session = f.findall('{http://www.nih.gov}readingSession')
        if not session:
            valid = False
        return valid

    # get the nodules as numpy arrays from annotation list
    @staticmethod
    def fill_nodule(nodule_list):
        annotations = np.zeros(shape=(512, 512))
        nodules = np.zeros(shape=(512, 512))
        vertices = []
        error = True
        for coord in nodule_list:
            # print((int(coord[1].text), int(coord[0].text)))
            vertices.append((int(coord[1].text), int(coord[0].text)))
            annotations[int(coord[1].text), int(coord[0].text)] = 1
            nodules[int(coord[1].text), int(coord[0].text)] = 1
        try:
            poly = Polygon(vertices)
            bnd = poly.bounds
            for x in range(int(bnd[0]), int(bnd[2] + 1)):
                for y in range(int(bnd[1]), int(bnd[3] + 1)):
                    point = Point(x, y)
                    if point.within(poly):
                        nodules[x, y] = 1
            error = False
        except ValueError:
            print('Polygone filling failed. Draw new nodule')
        return annotations, nodules, vertices, error

    # get nodule annotation
    def get_nodule_annotation(self, training_data = True):
        j = 0
        path = ''
        xml_path = ''
        path_list = []
        z_position = 0
        annotations = np.zeros(shape=(512, 512))
        nodules = np.zeros(shape=(512, 512))
        mel = 0
        while j < 1000:
            if training_data:
                xml_path = self.xml_training_list[randint(0, self.xml_training_list_length - 1)]
            else:
                xml_path = self.xml_eval_list[randint(0, self.xml_eval_list_length - 1)]

            if self.valid_xml(xml_path):
                f = ET.parse(xml_path).getroot()
                docs = f.findall('{http://www.nih.gov}readingSession')
                nodules = docs[randint(0, len(docs) - 1)].findall('{http://www.nih.gov}unblindedReadNodule')
                if len(nodules) > 0:
                    nod = nodules[randint(0, len(nodules) - 1)]
                    # get the annotation map
                    slices = nod.findall('{http://www.nih.gov}roi')
                    slice = slices[randint(0, len(slices) - 1)]
                    z_position = float(slice[0].text)
                    id = slice[1].text
                    if len(slice.findall('{http://www.nih.gov}edgeMap')) > 10:
                        # get melignancy
                        char = nod.findall('{http://www.nih.gov}characteristics')
                        if char:
                            mel = int(char[0].find('{http://www.nih.gov}malignancy').text)
                        else:
                            print('No melignancy info found!')
                        # read out annotation map of chosen nodule
                        annotations, nodules, vertices, error = LUNA.fill_nodule(slice.findall('{http://www.nih.gov}edgeMap'))
                        if not error:
                            j = 1000
            j = j + 1
        return xml_path, id, z_position, nodules, vertices, mel

    # normalizes image with houndsfield value cut
    @staticmethod
    def normalize(image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 800.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    @staticmethod
    def load_dcm(path):
        dc_file = dc.read_file(path)
        pic = dc_file.pixel_array
        pic = LUNA.normalize(pic)
        return pic

    # find dcm path of image that fits the z position and id of the chosen nodule
    @staticmethod
    def find_dcm_path(xml_path, id, z_position):
        k = -1
        while (not xml_path[k] == '/') and k > -1000:
            k = k - 1
        last_number = len(xml_path) + k
        cut_path = xml_path[0:last_number]
        path_list = ut.find('*dcm', cut_path)
        path = ''

        for im_path in path_list:
            dc_file = dc.read_file(im_path)
            image_z = (dc_file[0x0020, 0x0032].value)[2]
            image_id = dc_file[0x0008, 0x0018].value
            if image_id == id and image_z == z_position:
                path = im_path

        return path

    # find the centre of a given nodule
    @staticmethod
    def find_centre(vertices):
        # find the centre of the nodule
        x_min = 512
        x_max = 0
        y_min = 512
        y_max = 0
        for coord in vertices:
            if coord[0] < x_min:
                x_min = coord[0]
            if coord[0] > x_max:
                x_max = coord[0]
            if coord[1] < y_min:
                y_min = coord[1]
            if coord[1] > y_max:
                y_max = coord[1]
        x_cen = int((x_min + x_max) / 2)
        y_cen = int((y_min + y_max) / 2)
        return x_cen, y_cen

    # get location of a upper left corner for nodule and random cut
    @staticmethod
    def get_corner(vertices):
        x_cen, y_cen = LUNA.find_centre(vertices)
        j = 0
        upper_left = 0
        lower_right = 512
        while j < 100:
            centre_nod = [x_cen, y_cen] + np.random.randint(-20, 21, size=2)
            upper_left = centre_nod - 32
            lower_right = centre_nod + 32
            if upper_left[0] > 0 and upper_left[1] > 0 and lower_right[0] < 512 and lower_right[1] < 512:
                j = 100
            j = j + 1
        ul_ran= np.random.randint(150, 314, size=2)
        return upper_left, ul_ran

    # get a picture with nodules and malignancy
    def load_data(self, training_data = True):
        j = 0
        while j < 1000:
            xml_path, id, z_position, nodules, vertices, mel = self.get_nodule_annotation(training_data=training_data)
            path = self.find_dcm_path(xml_path, id, z_position)
            if not path == '':
                pic = self.load_dcm(path)
                j = 1000
            j = j+1
        ul_nod, ul_ran = LUNA.get_corner(vertices)
        return pic, nodules, ul_nod, ul_ran, mel

    # read out the x and y coordinate from xml file and return it as tuple
    @staticmethod
    def get_xy(knot):
        x = int(knot.find('x').text)
        y = int(knot.find('y').text)
        return (x,y)

    # loads specified image plus annotation from source file
    def load_from_source(self, id):
        document = ET.parse(self.source_path+'standard_eval.xml')
        root = document.getroot()
        nodule = root.find('.//nodule[@id="{}"]'.format(id))

        ver = nodule.find('vertices')
        annotations, nodules, vertices, error = LUNA.fill_nodule(ver.findall('edgeMap'))
        assert not error

        mel = int(nodule.find('mal').text)
        pic = self.load_dcm(nodule.find('dcm_path').text)
        ul_nod = LUNA.get_xy(nodule.find('ul_nod'))
        ul_ran = LUNA.get_xy(nodule.find('ul_ran'))
        return pic, nodules, ul_nod, ul_ran, mel

    # cuts out the 64^2 patch from given image with specified upper left corner coordinates
    @staticmethod
    def cut_data(pic, upper_left):
        pic_cut = pic[upper_left[0]:upper_left[0] + 64, upper_left[1]:upper_left[1] + 64]
        return pic_cut

