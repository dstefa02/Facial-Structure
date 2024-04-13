# encoding: utf-8

""" Retrieve Facial attributes and landmarks from Face++, Calculate several
Facial Ratios (fWHR, Cheekbone Prominence, Facial Symmetry), Preprocess the facial 
images (e.g. crop images), calculate 2 whole facial measures ("Pixel Intensity Measure" 
and "Facial Landmark Measure") using PCA+LDA (dimensionality reduction) and finally
store the results into a MongoDB.
"""
###############################################################################
# The following code is distributed under MIT license. Details are withheld
# for purposes of anonymity.
###############################################################################

###############################################################################
### Libraries ###
import requests, base64, json, time, sys, os, traceback, random, datetime, math
from pymongo import MongoClient
from queue import Queue
from threading import Thread
import threading
from PIL import Image
import os, sys, datetime, re, time, json, traceback, math
from pymongo import MongoClient
from os import path
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import shapely.geometry as geometry
from shapely.geometry import Point, Polygon
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, classification_report
from sklearn.metrics import accuracy_score, roc_curve, classification_report
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.stats import mannwhitneyu
###############################################################################
### Global Variables ###

# Max requests per minute (defined by Face++ API)
FACEPLPL_MAX_QUERY_PER_SECOND = 1.0

### Face++ Detect API ###
# Face++ Detect API URL
FACEPLPL_DETECT_API = 'https://api-us.faceplusplus.com/facepp/v3/detect' 
# Face++ Detect API Data for Post
faceplpl_detect_api_data = {
    'api_key': 'INSERT_API_KEY',
    'api_secret': 'INSERT_API_SECRET',
    'return_landmark': '2', # '1': return 106-point landmarks, '2': return 83-point landmarks
    'return_attributes': 'gender,age,smiling,headpose,facequality,blur,eyestatus,' +
        'emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus'            
}

### Face++ Dense API ###
# Face++ Dense API for 1000 landmarks
FACEPLPL_DENSE_FACIAL_LANDMARK_API = 'https://api-us.faceplusplus.com/facepp/v1/face/thousandlandmark'
# Face++ Dense API Data for Post
faceplpl_dense_api_data = {
    'api_key': 'INSERT_API_KEY',
    'api_secret': 'INSERT_API_SECRET',
    'return_landmark': 'all'    
}

### Face++ API Keys ###
# Config data for Face++ e.g. api key, api secret
# Note: You can use more than 1 API key for the Face++ API
CONFIG_FACEPLPL_ACCOUNTS = {
    "accounts": [
        {   
            "api_key": "API-KEY-1",
            "api_secret": "API-SECRET-1"
        },
        {   
            "api_key": "API-KEY-2",
            "api_secret": "API-SECRET-2"
        }
    ]
}

# Stored connections with database (MongoDB) for each thread
db_connections = []
# The queue for tasks coordination between threads
queue = Queue()
# The number of threads for parallel requests
NUM_THREADS = 2

# Log file for storing errors
error_log_file = open('LOG-ERROR_Analyze_Crunchbase_Image_Face++.log', 'w')
# Log file for storing the progress
progress_log_file = open('LOG-PROGRESS_Analyze_Crunchbase_Image_Face++.log', 'w')

# Directory to read initial profile images
INITIAL_PROFILE_IMAGES_DIR = "Crunchbase_Profile_Images"

# Cropped Images Dir
CROPPED_IMAGES_DIR = 'Images/Cropped_Images/'
# Grayscales Images Dir
GRAYSCALE_IMAGES_DIR = 'Images/Grayscale_Images/'
# Resized Images Dir
RESIZED_IMAGES_DIR = 'Images/Resized_Images/'
# Padded with extra black pixels Images Dir
PADDED_IMAGES_DIR = 'Images/Padded_Images/'
# Without Background Images Dir
BACKGROUND_IMAGES_DIR = 'Images/Background_Images/'
# Without Background Images Dir
EQ_HIST_IMAGES_DIR = 'Images/Output_Images_EQ_Hist_Images_ENTRE_VS_NONENTRE_5D/'

# Database ip
DATABASE_IP = 'localhost'
# Database port
DATABASE_PORT = 27017
# Database name for storing Face++ data
DATABASE_NAME = 'Faces'
# Collection name for storing Face++ results
DATABASE_FACE_COLLECTION = 'CB_Entre_and_NonEntre'

###############################################################################
def faceplpl_request(input_dir, api_url, faces_progress_var):
    """ Retrieve the attributes and landmarks from Face++ by sending a list of photos.

    Args:
        input_dir (str): The name of the directory with the profile images
        api_url (str): The Face++ URL for a specific endpoint
        faces_progress_var (str): The db field in which will be stored the retrieved
                                  attributes and landmarks
    Returns:
        -
    """

    print('Input dir for images: ' + input_dir, '\nFACE++ API URL: ' + api_url)
    print('='*50)

    # Find all the names of the profile images in the directory
    images_dict = dict()
    for image_file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, image_file)):

            fields = image_file.split('.')
            person_uuid = fields[0]
            processed_person_uuid = person_uuid.replace('-', '')
            ext = fields[1]

            # find size of images
            im = Image.open(input_dir+image_file)
            width, height = im.size

            images_dict[processed_person_uuid] = {
                'image_file': image_file,
                'image_size': {'width': width, 'height': height},
            }

    # Find all stored faces from database
    stored_faces_dict = dict()

    client = MongoClient(DATABASE_IP, DATABASE_PORT)
    collection_faces = client[DATABASE_NAME][DATABASE_FACE_COLLECTION]

    query = {str(faces_progress_var+'_msg'): {'$exists': 1}}
    select = {'_id': 1, str(faces_progress_var+'_msg'): 1}

    for obj in collection_faces.find(query, select):
        if (obj[str(faces_progress_var+'_msg')] == 'SUCCESS' or 
            obj[str(faces_progress_var+'_msg')] == 'INVALID_IMAGE_SIZE: image_file'):
            stored_faces_dict[obj['_id']] = 1


    # Initialize threads
    for thread_id in range(NUM_THREADS):
        #connect to db
        client = MongoClient(DATABASE_IP, DATABASE_PORT)
        collection = client[DATABASE_NAME][DATABASE_FACE_COLLECTION]
        db_connections.append(collection)
        # start threads
        worker = Thread(target=thread_faceplusplus_request, args=(thread_id, queue))
        worker.setDaemon(True)
        worker.start()


    # Start sending images to different threads
    count = 1
    for item_key in images_dict:

        image_path = input_dir + '/' + images_dict[item_key]['image_file']
        uuid = images_dict[item_key]['image_file'].split('.', 1)[0]

        if not uuid in stored_faces_dict:

            # Rotate key in each job/task
            cur_key = CONFIG_FACEPLPL_ACCOUNTS['accounts'][0]['api_key']
            cur_secret = CONFIG_FACEPLPL_ACCOUNTS['accounts'][0]['api_secret']
            if count % 2 == 0: # 2=number of api keys - round robin
                cur_key = CONFIG_FACEPLPL_ACCOUNTS['accounts'][1]['api_key']
                cur_secret = CONFIG_FACEPLPL_ACCOUNTS['accounts'][1]['api_secret']

            # Put new job/task into queue
            queue.put([
                uuid, image_path, count, images_dict[item_key], 
                faceplpl_detect_api_data, FACEPLPL_DETECT_API, 
                cur_key, cur_secret, faces_progress_var, str(faces_progress_var+'_msg')
            ])

            # sleep for some seconds
            time.sleep(FACEPLPL_MAX_QUERY_PER_SECOND)

        else: # Image already processed
            progress_log_file.write('Image ' + str(count) + ' Already Exist    ' + 
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '\n')
            progress_log_file.flush()
        count += 1


    # Block until all items in the queue have been gotten and processed.
    queue.join()

    progress_log_file.write('\nFinished processing images\n' + '='*50 + '\n')
###############################################################################
def thread_faceplusplus_request(thread_id, queue):
    """A multithreaded function that sends each photo to the face++ API, 
       receives the results and stores them in the MongoDB.

    Args:
        thread_id (int): The id of each thread
        queue (Queue): The queue for tasks coordination between threads

    Returns:
        -
    """

    while True:
        item = queue.get()

        uuid = item[0]
        image_path = item[1]
        image_number = item[2]
        images_item = item[3]
        faceplpl_api_data = item[4]
        faceplpl_api_url = item[5]
        api_key = item[6]
        api_secret = item[7]
        new_field_db = item[8]
        new_field_msg_db = item[9]

        collection = db_connections[thread_id]

        # insert api key and secret
        faceplpl_api_data['api_key']= api_key
        faceplpl_api_data['api_secret']= api_secret

        # Send request to face++
        r = requests.post(faceplpl_api_url, 
            data=faceplpl_api_data, files={'image_file': open(image_path, 'rb')})

        try:
            data = json.loads(r.text)
            faceplusplus_json = ''

            if 'faces' in data:
                faceplusplus_json = data['faces']
            elif 'face' in data:
                faceplusplus_json = data['face']
            else:
                raise ValueError('Cannot find face or faces key in dict!')

            # Update doc in MongoDB
            db_res = collection.update(
                {'_id' : uuid}, 
                {'$set': 
                    {
                        'image_size': images_item['image_size'],
                        new_field_db: faceplusplus_json, 
                        new_field_msg_db: 'SUCCESS'
                    }
                }, upsert=True)


            progress_log_file.write('Image ' + str(image_number) + 
                ' Inserted Successfully    ' + 
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + '\n')
            progress_log_file.flush()
            
        except Exception as e:
            if ('error_message' in data and data['error_message'] == 
                'CONCURRENCY_LIMIT_EXCEEDED'):

                queue.put(item)
            elif 'error_message' in data:

                db_res = collection.update(
                    {'_id' : uuid}, 
                    {'$set': 
                        {
                            'image_size': images_item['image_size'],
                            new_field_msg_db: data['error_message']
                        }
                    }, upsert=True)

                error_log_file.write('='*50)
                error_log_file.write('\n' + 'Image number: ' + str(image_number))
                error_log_file.write('\nFace++ error message: ' + 
                    data['error_message'] + '\n')
                error_log_file.write('='*50)
                error_log_file.flush()
            else:
                error_log_file.write('\n' + str(data))
                err = str(traceback.format_exc())
                err = err.replace('Traceback (most recent call last):', 'ERROR:')
                error_log_file.write('\n' + err)
                error_log_file.flush()

        queue.task_done()
###############################################################################

###############################################################################
def facial_ratios():
    """Calculate 3 facial ratios (fWHR, cheekbone_prominence, facial symmetry)
       and store them in the MongoDB

    Args:
        -
    Returns:
        -
    """

    print('Calculate all the facial ratios and store them into MongoDB')
    print('='*50)

    # Connect to MongoDB
    client = MongoClient(DATABASE_IP, DATABASE_PORT)
    collection_faces = client[DATABASE_NAME][DATABASE_FACE_COLLECTION]

    query = {'faceplpl_detect': {'$exists': 1}}
    select = {'_id': 1, 'faceplpl_detect': 1}

    for obj in collection_faces.find(query, select):

        if len(obj['faceplpl_detect']) != 1:
            continue

        # head pose
        yaw = obj['faceplpl_detect'][0]['attributes']['headpose']['yaw_angle']
        pitch = obj['faceplpl_detect'][0]['attributes']['headpose']['pitch_angle']
        roll = obj['faceplpl_detect'][0]['attributes']['headpose']['roll_angle']

        if (yaw > 5 or yaw < -5 or 
            pitch > 5 or pitch < -5 or
            roll > 5 or roll < -5):
            continue

        # fWHR
        fWHR = calc_fWHR(obj['faceplpl_detect'][0]['landmark'])

        # fWHR - lower
        # fWHR_lower = calc_fWHR_lower(obj['faceplpl_detect'][0]['landmark'])

        # Cheekbone prominence
        cheekbone_prominence = calc_cheekbone_prominence(obj['faceplpl_detect'][0]['landmark'])

        # OFA, CFA - Horizontal Facial asymmetry
        OFA, CFA = calculate_horizontal_asymmetry(obj['faceplpl_detect'][0]['landmark'])

        # Store to db
        db_res = collection_faces.update(
            {'_id' : obj['_id']}, 
            {'$set': 
                {
                    'facial_ratios': {
                        'fWHR': fWHR, 
                        'Cheekbone_prominence': cheekbone_prominence, 
                        'OFA': OFA, 
                        'CFA': CFA
                    },
                }
            }
        )
###############################################################################
def calc_fWHR(landmarks):
    """Calculate fWHR

    Args:
        landmarks (dict): A dictionary that contains the landmarks as retrieved
                          by Face++.
    Returns:
        float: The calculated fWHR measure
    """

    contour_left_x = landmarks['contour_left1']['x']
    contour_left_y = landmarks['contour_left1']['y']
    contour_right_x = landmarks['contour_right1']['x']
    contour_right_y = landmarks['contour_right1']['y']
    upper_lip_top_x = landmarks['mouth_upper_lip_top']['x']
    upper_lip_top_y = landmarks['mouth_upper_lip_top']['y']
    left_eyebrow_x = landmarks['left_eyebrow_right_corner']['x']
    left_eyebrow_y = landmarks['left_eyebrow_right_corner']['y']
    right_eyebrow_x = landmarks['right_eyebrow_left_corner']['x']
    right_eyebrow_y = landmarks['right_eyebrow_left_corner']['y']

    # Calculating eyebrow mid point.
    mid_eyebrow_x = (left_eyebrow_x + right_eyebrow_x)/2
    mid_eyebrow_y = (left_eyebrow_y + right_eyebrow_y)/2 

    # Calculating face height by using the distance between two points formula.
    partX = math.pow((upper_lip_top_x - mid_eyebrow_x), 2)
    partY = math.pow((upper_lip_top_y - mid_eyebrow_y), 2)
    height = math.sqrt(partY+partX)

    # Calculating face width by using the distance between two points formula.
    partX = math.pow((contour_left_x - contour_right_x), 2)
    partY = math.pow((contour_left_y - contour_right_y), 2)
    width = math.sqrt(partY+partX)

    # Calculating FHWR.
    ratio = width / height

    return ratio
###############################################################################
def calc_fWHR_lower(landmarks):
    """Calculate fWHR-lower

    Args:
        landmarks (dict): A dictionary that contains the landmarks as retrieved
                          by Face++.
    Returns:
        float: The calculated fWHR-lower measure
    """

    contour_left_x = landmarks['contour_left1']['x']
    contour_left_y = landmarks['contour_left1']['y']
    contour_right_x = landmarks['contour_right1']['x']
    contour_right_y = landmarks['contour_right1']['y']
    contour_chin_x = landmarks['contour_chin']['x']
    contour_chin_y = landmarks['contour_chin']['y']
    left_eye_x = landmarks['left_eye_right_corner']['x']
    left_eye_y = landmarks['left_eye_right_corner']['y']
    right_eye_x = landmarks['right_eye_left_corner']['x']
    right_eye_y = landmarks['right_eye_left_corner']['y']

    # Calculating eye mid point.
    mid_eye_x = (left_eye_x + right_eye_x)/2
    mid_eye_y = (left_eye_y + right_eye_y)/2 

    # Calculating face height by using the distance between two points formula.
    partX = math.pow((contour_chin_x - mid_eye_x), 2)
    partY = math.pow((contour_chin_y - mid_eye_y), 2)
    height = math.sqrt(partY+partX)

    # Calculating face width by using the distance between two points formula.
    partX = math.pow((contour_left_x - contour_right_x), 2)
    partY = math.pow((contour_left_y - contour_right_y), 2)
    width = math.sqrt(partY+partX)

    # Calculating FHWR.
    ratio = width / height

    return ratio
###############################################################################
def calc_cheekbone_prominence(landmarks):
    """Calculate Cheekbone Prominence

    Args:
        landmarks (dict): A dictionary that contains the landmarks as retrieved
                          by Face++.
    Returns:
        float: The calculated Cheekbone Prominence measure
    """

    contour_leftA_x = landmarks['contour_left1']['x']
    contour_leftA_y = landmarks['contour_left1']['y']
    contour_rightA_x = landmarks['contour_right1']['x']
    contour_rightA_y = landmarks['contour_right1']['y']
    contour_leftB_x = landmarks['contour_left5']['x']
    contour_leftB_y = landmarks['contour_left5']['y']
    contour_rightB_x = landmarks['contour_right5']['x']
    contour_rightB_y = landmarks['contour_right5']['y'] 

    # Calculating top face width by using the distance between two points formula.
    partX = math.pow((contour_leftA_x - contour_rightA_x), 2)
    partY = math.pow((contour_leftA_y - contour_rightA_y), 2)
    widthA = math.sqrt(partY+partX)

    # Calculating top face width by using the distance between two points formula.
    partX = math.pow((contour_leftB_x - contour_rightB_x), 2)
    partY = math.pow((contour_leftB_y - contour_rightB_y), 2)
    widthB = math.sqrt(partY+partX)

    # Calculating FHWR.
    ratio = widthA / widthB

    return ratio
###############################################################################
def calculate_horizontal_asymmetry(landmarks):
    """Calculate facial symmetry (OFA and CFA)

    Args:
        landmarks (dict): A dictionary that contains the landmarks as retrieved
                          by Face++.
    Returns:
        float: The calculated OFA measure
        float: The calculated CFA measure
    """

    # Find points/landmarks
    facial_symmetry_points = find_facial_symmetry_points(landmarks)

    # Calculate midpoints of D lines
    midpoints = calc_midpoints(facial_symmetry_points)

    # Calculate overall facial asymmetry
    FA_adjusted_with_max_min = calc_overall_facial_symmetry(midpoints)

    # Calculate central facial asymmetry
    CFA_adjusted_with_max_min = calc_central_facial_symmetry(midpoints)

    return FA_adjusted_with_max_min, CFA_adjusted_with_max_min
###############################################################################
def find_facial_symmetry_points(landmarks):
    """Find the correct landmarks for facial symmetry

    Args:
        landmarks (dict): A dictionary that contains the landmarks as retrieved
                          by Face++.
    Returns:
        dict: A dictionary with the facial landmarks
    """

    facial_symmetry_points = {}

    contour_left = [
        landmarks['contour_left1'], landmarks['contour_left2'],
        landmarks['contour_left3'], landmarks['contour_left4'],
        landmarks['contour_left5'], landmarks['contour_left6'],
        landmarks['contour_left7'], landmarks['contour_left8'],
        landmarks['contour_left9']
    ]
    contour_right = [
        landmarks['contour_right1'], landmarks['contour_right2'],
        landmarks['contour_right3'], landmarks['contour_right4'],
        landmarks['contour_right5'], landmarks['contour_right6'],
        landmarks['contour_right7'], landmarks['contour_right8'],
        landmarks['contour_right9']
    ]

    # left_eye_left_corner, right_eye_right_corner
    P1_x  = landmarks['left_eye_left_corner']['x']
    P2_x  = landmarks['right_eye_right_corner']['x']

    # left_eye_right_corner, right_eye_left_corner
    P3_x  = landmarks['left_eye_right_corner']['x']
    P4_x  = landmarks['right_eye_left_corner']['x']

    # contour_left2, contour_right2
    P5_x  = landmarks['contour_left2']['x']
    P6_x  = landmarks['contour_right2']['x']

    # nose_left, nose_right
    P7_x  = landmarks['nose_left']['x']
    P8_x  = landmarks['nose_right']['x']

    # mouth_left_corner, mouth_right_corner
    P11_x = landmarks['mouth_left_corner']['x']
    mouth_left_corner_y = landmarks['mouth_left_corner']['y']
    P12_x = landmarks['mouth_right_corner']['x']
    mouth_right_corner_y = landmarks['mouth_right_corner']['y']

    # contour_chin
    P13_x = landmarks['contour_chin']['x']


    # Jaw width was measured as face width at the y coordinate of the mouth corners (P11 and P12)
    closest_to_P9_left = ''
    diff_to_P9_left_point = 100000
    closest_to_P10_right = ''
    diff_to_P10_right_point = 100000


    # P9 = closest y of contour_left and mouth_left_corner (P11)
    # P10 = closest y of contour_right and mouth_right_corner (P12)
    P9_x = 0
    P10_x = 0

    for i in range (1, 10):
        left_point_y = landmarks['contour_left'+str(i)]['y']
        right_point_y = landmarks['contour_right'+str(i)]['y']

        if abs(left_point_y - mouth_left_corner_y) < diff_to_P9_left_point:
            diff_to_P9_left_point = abs(left_point_y - mouth_left_corner_y)
            closest_to_P9_left = str(i)

        if abs(right_point_y - mouth_right_corner_y) < diff_to_P10_right_point:
            diff_to_P10_right_point = abs(right_point_y - mouth_right_corner_y)
            closest_to_P10_right = str(i)

    P9_x = landmarks['contour_left'+str(closest_to_P9_left)]['x']
    P10_x = landmarks['contour_right'+str(closest_to_P10_right)]['x']

    # Set the final landmarks for facial symmetry
    facial_symmetry_points['P1_x']  = P1_x
    facial_symmetry_points['P2_x']  = P2_x
    facial_symmetry_points['P3_x']  = P3_x
    facial_symmetry_points['P4_x']  = P4_x
    facial_symmetry_points['P5_x']  = P5_x
    facial_symmetry_points['P6_x']  = P6_x
    facial_symmetry_points['P7_x']  = P7_x
    facial_symmetry_points['P8_x']  = P8_x
    facial_symmetry_points['P9_x']  = P9_x
    facial_symmetry_points['P10_x'] = P10_x
    facial_symmetry_points['P11_x'] = P11_x
    facial_symmetry_points['P12_x'] = P12_x
    facial_symmetry_points['P13_x'] = P13_x

    facial_symmetry_points['P1_y']  = landmarks['left_eye_left_corner']['y']
    facial_symmetry_points['P2_y']  = landmarks['right_eye_right_corner']['y']
    facial_symmetry_points['P3_y']  = landmarks['left_eye_right_corner']['y']
    facial_symmetry_points['P4_y']  = landmarks['right_eye_left_corner']['y']
    facial_symmetry_points['P5_y']  = landmarks['contour_left2']['y']
    facial_symmetry_points['P6_y']  = landmarks['contour_right2']['y']
    facial_symmetry_points['P7_y']  = landmarks['nose_left']['y']
    facial_symmetry_points['P8_y']  = landmarks['nose_right']['y']
    facial_symmetry_points['P9_y']  = landmarks['contour_left'+str(closest_to_P9_left)]['y']
    facial_symmetry_points['P10_y'] = landmarks['contour_right'+str(closest_to_P10_right)]['y']
    facial_symmetry_points['P11_y'] = landmarks['mouth_left_corner']['y']
    facial_symmetry_points['P12_y'] = landmarks['mouth_right_corner']['y']
    facial_symmetry_points['P13_y'] = landmarks['contour_chin']['y']

    return facial_symmetry_points
###############################################################################
def calc_midpoints(points):
    """Calculate the midpoints for each pair of landmarks.

    Args:
        points (dict): The dictionary with the landmarks.
    Returns:
        dict: A dictionary with the midpoints
    """

    midpoints = {}
    midpoints['D1'] = ((points['P1_x'] - points['P2_x']) / 2.0) + points['P2_x']
    midpoints['D2'] = ((points['P3_x'] - points['P4_x']) / 2.0) + points['P4_x']
    midpoints['D3'] = ((points['P5_x'] - points['P6_x']) / 2.0) + points['P6_x']
    midpoints['D4'] = ((points['P7_x'] - points['P8_x']) / 2.0) + points['P8_x']
    midpoints['D5'] = ((points['P11_x'] - points['P12_x']) / 2.0) + points['P12_x']
    midpoints['D6'] = ((points['P9_x'] - points['P10_x']) / 2.0) + points['P10_x']

    return midpoints
###############################################################################
def calc_overall_facial_symmetry(midpoints):
    """Calculate the Overall facial symmetry

    Args:
        landmarks (dict): A dictionary that contains the calculated midpoints
    Returns:
        float: The calculated Overall facial symmetry measure
    """

    total_diff_adjusted_with_max_min = 0

    for i in range(1,6):
        key_1 = 'D' + str(i)

        for j in range(i+1,6+1):
            key_2 = 'D' + str(j)

            max_midpoint = max(midpoints[key_1], midpoints[key_2])
            min_midpoint = min(midpoints[key_1], midpoints[key_2])
            total_diff_adjusted_with_max_min += min_midpoint*1.0/max_midpoint

    FA_adjusted_with_max_min = 15 - total_diff_adjusted_with_max_min

    return FA_adjusted_with_max_min
###############################################################################
def calc_central_facial_symmetry(midpoints):
    """Calculate the Central facial symmetry

    Args:
        landmarks (dict): A dictionary that contains the calculated midpoints
    Returns:
        float: The calculated Central facial symmetry measure
    """

    total_diff_adjusted_with_max_min = 0

    for i in range(1,6):
        key_1 = 'D' + str(i)
        key_2 = 'D' + str(i+1)

        max_midpoint = max(midpoints[key_1], midpoints[key_2])
        min_midpoint = min(midpoints[key_1], midpoints[key_2])
        total_diff_adjusted_with_max_min += min_midpoint*1.0/max_midpoint

    CFA_adjusted_with_max_min = 5 - total_diff_adjusted_with_max_min

    return CFA_adjusted_with_max_min
###############################################################################

###############################################################################
def get_landmarks_from_db(db_collection, landmarks_column_name, progress_column_name):
    """Retrieve the stored landmarks from MondoDB

    Args:
        db_collection (str): The collection that contains the facial related attributes
        landmarks_column_name (str): The column name that contains the landmarks
        progress_column_name (str): The column name that shows whether the landmarks have been
                                    downloaded successfully.
    Returns:
        dict: A dictionary with the landmarks of each person
    """

    print('Start: Get landmarks from DB')

    # Find all stored faces from database
    client = MongoClient(DATABASE_IP, DATABASE_PORT)
    collection_faces = client[DATABASE_NAME][db_collection]
    faces_dict = dict()

    for obj in collection_faces.find({progress_column_name: {'$exists': 1}}):
        if obj[progress_column_name] == 'SUCCESS':

            if not 'landmark' in obj[landmarks_column_name]:
                continue

            # if points is out of bounds change it
            for face_area in obj[landmarks_column_name]['landmark']:
                for point in obj[landmarks_column_name]['landmark'][face_area]:
                    point_item = obj[landmarks_column_name]['landmark'][face_area][point]

                    if type(point_item) == dict:
                        if obj[landmarks_column_name]['landmark'][face_area][point]['x'] < 0:
                            obj[landmarks_column_name]['landmark'][face_area][point]['x'] = 0
                        if obj[landmarks_column_name]['landmark'][face_area][point]['x'] > 100:
                            obj[landmarks_column_name]['landmark'][face_area][point]['x'] = 100
                        if obj[landmarks_column_name]['landmark'][face_area][point]['y'] < 0:
                            obj[landmarks_column_name]['landmark'][face_area][point]['y'] = 0
                        if obj[landmarks_column_name]['landmark'][face_area][point]['y'] > 180:
                            obj[landmarks_column_name]['landmark'][face_area][point]['y'] = 180

            faces_dict[obj['_id']] = obj

    print('Finish: Get landmarks from DB\n')
    return faces_dict
###############################################################################
def find_images_filenames(path_dir):
    """Find the names of the images that stored in a specific directory.

    Args:
        path_dir (str): The path of the directory that contains profile pictures.

    Returns:
        list: A list with the filenames
    """

    # Find all the names of the profile images in the directory
    images_filenames = []
    for image_f in os.listdir(path_dir):
        if os.path.isfile(os.path.join(path_dir, image_f)):
            images_filenames.append({'name': image_f, 'path': path_dir+image_f})

    return images_filenames
###############################################################################
def find_outermost_face_points(landmarks_dict, image_height, image_width):
    """Find the outermost landmarks in a face in 4 directions: x, -x, y and -y.

    Args:
        landmarks_dict (dict): The dictionary with the landmarks related to an image
        image_height (float): The height of an image
        image_width (float): The width of an image

    Returns:
        dict: A dictionary with the outermost coordinates in a face.
    """

    outermost_points = {
        'min_x': '',
        'max_x': '',
        'min_y': '',
        'max_y': '',
    }

    # Layer 1: left_eye, left_eyebrow, mouth, face, right_eyebrow, right_eye, left_eye_eyelid, right_eye_eyelid, nose
    for group_item in landmarks_dict:

        # e.g. left_eye_pupil_radius, left_eye_8, left_eye_26
        for point in landmarks_dict[group_item]:
            point = landmarks_dict[group_item][point]

            if type(point) == dict:
                if (outermost_points['min_x'] == '' and outermost_points['max_x'] == '' and
                    outermost_points['min_y'] == '' and outermost_points['max_y'] == ''):

                    outermost_points['min_x'] = point['x']
                    outermost_points['max_x'] = point['x']
                    outermost_points['min_y'] = point['y']
                    outermost_points['max_y'] = point['y']
                else:
                    if point['x'] < outermost_points['min_x']:
                        outermost_points['min_x'] = point['x']
                    if point['x'] > outermost_points['max_x']:
                        outermost_points['max_x'] = point['x']

                    if point['y'] < outermost_points['min_y']:
                        outermost_points['min_y'] = point['y']
                    if point['y'] > outermost_points['max_y']:
                        outermost_points['max_y'] = point['y']

    # Add extra padding of +-2 pixels
    if outermost_points['min_x'] - 2 >= 0:
        outermost_points['min_x'] -= 2
    if outermost_points['min_y'] - 2 >= 0:
        outermost_points['min_y'] -= 2
    if outermost_points['max_x'] + 2 <= image_width:
        outermost_points['max_x'] += 2
    if outermost_points['max_y'] + 2 <= image_height:
        outermost_points['max_y'] += 2

    # Check for weird values and replace them
    if outermost_points['min_x'] < 0:
        outermost_points['min_x'] = 0
    if outermost_points['min_y'] < 0:
        outermost_points['min_y'] = 0
    if outermost_points['max_x'] > image_width:
        outermost_points['max_x'] = image_width
    if outermost_points['max_y'] > image_height:
        outermost_points['max_y'] = image_height

    return outermost_points
###############################################################################
def crop_images(input_images_dir, output_images_dir):
    """Crop images based on their outermost landmarks.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.

    Returns:
        -
    """

    print('Start Cropping images of folder:', input_images_dir)

    # Retrieve the stored landmarks from MongoDB
    faces_dict = get_landmarks_from_db(DATABASE_FACE_COLLECTION, 
        'faceplpl_dense_v1', 'faceplpl_dense_v1_msg')

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    ### Read images from local folder ###
    images_filenames = find_images_filenames(input_images_dir)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        if uuid in faces_dict:
            print(uuid)
            # Read/Open image
            img = cv2.imread(item_image['path'])

            # Get size of image
            height, width, depth = img.shape
            print('Initial image size:', img.shape)

            # Find outermost_points in face
            outermost_points = find_outermost_face_points(
                faces_dict[uuid]['faceplpl_dense_v1']['landmark'], height, width
            )

            # Crop image based on the outermost points in face
            cropped_img = img[
                    outermost_points['min_y']:outermost_points['max_y'], 
                    outermost_points['min_x']:outermost_points['max_x']
                ]

            print(outermost_points)
            print('New Cropped image size:', cropped_img.shape)
            print('='*50)
            cv2.imwrite(output_images_dir+item_image['name'], cropped_img)

    print('Finish Cropping images')
###############################################################################
def convert_to_grayscale_images(input_images_dir, output_images_dir):
    """Convert images to grayscale.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.

    Returns:
        -
    """

    # luminance is by far more important in distinguishing visual features than color
    print('Start Grayscaling images of folder:', input_images_dir)

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    ### Read images from local folder ###
    images_filenames = find_images_filenames(input_images_dir)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read/Open image
        img = cv2.imread(item_image['path'])

        # Grayscale pixel values range from 0 (for black) to 255 (for white)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('Original image', img)
        # cv2.imshow('Gray image', gray_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(output_images_dir+item_image['name'], gray_img)

    print('Finish Grayscaling images')
###############################################################################
def resize_images(input_images_dir, output_images_dir, new_width=100, inter=cv2.INTER_LANCZOS4):
    """Resize images based on the given new_width.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.
        new_width (float): The new width of all images.
        inter (str): The method for the image interpolation

    Returns:
        -
    """

    print('Start Rescaling images of folder:', input_images_dir)

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    ### Read images from local folder ###
    images_filenames = find_images_filenames(input_images_dir)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read image
        img = cv2.imread(item_image['path'])

        # initialize the dimensions of the image and grab the image size
        dim = None
        (old_height, old_width) = img.shape[:2]

        # calculate the ratio of the width and construct the dimensions
        ratio_w = new_width / float(old_width)
        dim = (new_width, int(old_height * ratio_w))

        # Resize the image
        resized_img = cv2.resize(img, dim, interpolation=inter)

        # Check if height is smaller than 100 and resize again but based on height
        (old_height, old_width) = resized_img.shape[:2]
        if old_height < 100:
            new_height = 100
            # calculate the ratio of the height and construct the dimensions
            ratio_h = new_height / float(old_height)
            dim = (new_height, int(old_width * ratio_h))

            # Resize the image
            resized_img = cv2.resize(resized_img, dim, interpolation=inter)

        cv2.imwrite(output_images_dir+item_image['name'], resized_img)

    print('Finish Rescaling images')
###############################################################################
def pad_images(input_images_dir, output_images_dir):
    """Pad images with extra black pixels.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.

    Returns:
        -
    """

    print('Start Pad with extra space images of folder:', input_images_dir)

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    ### Read images from local folder ###
    images_filenames = find_images_filenames(input_images_dir)

    ### Find larger image in terms of height
    max_height = 0
    max_width = 0
    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read/Open image
        img = cv2.imread(item_image['path'])
        # Get size of image
        height, width, depth = img.shape

        # Find max height
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    print('Max width:', max_width)
    print('Max height:', max_height)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read/Open image
        img = cv2.imread(item_image['path'])
        # Get size of image
        height, width, depth = img.shape

        # Calculate how much extra padding is needed
        pad_height_pixels = max_height - height

        # Pad with extra black pixels
        # top, bottom, left, right - border width in number of pixels in corresponding directions
        padded_img = cv2.copyMakeBorder(img, pad_height_pixels, 0, 0, 0, cv2.BORDER_CONSTANT)

        # cv2.imshow('Original image', img)
        # cv2.imshow('Padded image', padded_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(output_images_dir+item_image['name'], padded_img)

    print('Finish Pad with extra space images')
###############################################################################
def remove_background(input_images_dir, output_images_dir):
    """Remove the background from Facial images.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.

    Returns:
        -
    """

    print('Start Remove Background images of folder:', input_images_dir)

    # Retrieve the updated landmarks from MongoDB
    updated_faces_dict = get_landmarks_from_db(DATABASE_FACE_COLLECTION, 
        'faceplpl_dense_v2', 'faceplpl_dense_v2_msg') 

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    for item in updated_faces_dict:

        coords = []
        nodes = []

        face = updated_faces_dict[item]['faceplpl_dense_v2']['landmark']
        for key1 in face['face']:
            nodes.append((face['face'][key1]['x'], face['face'][key1]['y']))

        # Connect points into a polygon based on euclidean distance
        cur_point = nodes[0]
        coords.append(cur_point)
        nodes.pop(0)

        while len(nodes) > 0: 
            index = distance.cdist([cur_point], nodes).argmin()
            coords.append(nodes[index])
            cur_point = nodes[index]
            nodes.pop(index)

        # Create polygon
        updated_faces_dict[item]['poly'] = Polygon(coords)


    ### Read images from local folder ###
    images_filenames = find_images_filenames(input_images_dir)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read/Open image
        img = cv2.imread(item_image['path'])

        # grab the image dimensions
        h = img.shape[0]
        w = img.shape[1]
        
        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):

                p1 = Point(x, y)

                if not p1.within(updated_faces_dict[uuid]['poly']):
                    # threshold the pixel
                    img[y, x] = 0

        cv2.imwrite(output_images_dir+item_image['name'], img)

    print('Finish Remove Background images')
###############################################################################
def histogram_equalization(input_images_dir, output_images_dir):
    """Apply Histogram Equalization to a set of images.

    Args:
        input_images_dir (str): The directory with the initial input profile images.
        output_images_dir (str): The directory with the processed profile images.

    Returns:
        -
    """

    print('Start Histogram Equalization images of folder:', input_images_dir)

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)
    
    # Read images from local folder
    images_filenames = find_images_filenames(input_images_dir)

    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        # Read image
        img = cv2.imread(item_image['path'])

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Histogram Equalization
        eq_hist_img = cv2.equalizeHist(grayimg)

        cv2.imwrite(output_images_dir+item_image['name'], eq_hist_img)

    print('Finish Histogram Equalization images')
###############################################################################

###############################################################################
def read_input_datasets():
    """Read the stored dataset of entrepreneurs and non-entrepreneurs from a tsv file.

    Args:
        -

    Returns:
        dict: A dictionary with the independent and dependent variables for each person.
    """

    # Read new input files
    input_f = open('Input_Dataset/OUTPUT_Expanded_Entrepreneurship_5_degrees_08_09_2020.tsv', 'r')
    header_1 = next(input_f).strip("\n") # skip header
    header_fields = header_1.split("\t")
    people_input_dict = dict()
    for line in input_f:
        fields = line.strip("\n").split("\t")
        uuid = fields[0].replace("-", "")

        people_input_dict[uuid] = {}
        people_input_dict[uuid]['new_has_received_funding'] = ''
        people_input_dict[uuid]['new_total_received_funding_usd'] = ''
        people_input_dict[uuid]['processed_revenue'] = ''
        people_input_dict[uuid]['processed_most_recent_valuation'] = ''
        people_input_dict[uuid]['dataset'] = '0'

        for i in range (0, len(header_fields)):
            people_input_dict[uuid][header_fields[i]] = fields[i]
    input_f.close()

    input_f = open('Input_Dataset/OUTPUT_Expanded_Success_5_degrees_08_09_2020.tsv', 'r')
    header_2 = next(input_f).strip("\n") # skip header
    header_fields = header_2.split("\t")
    for line in input_f:
        fields = line.strip("\n").split("\t")
        uuid = fields[0].replace("-", "")
        people_input_dict[uuid]['dataset'] = '1'

        for i in range (0, len(header_fields)):
            # if not header_fields[i] in people_input_dict[uuid]:
            people_input_dict[uuid][header_fields[i]] = fields[i]

    input_f.close()

    return people_input_dict
###############################################################################
def pixels_and_landmarks_to_1D_array(images_input_dir):
    """Convert the landmarks and pixels vectors into 1D vectors.

    Args:
        input_images_dir (str): The directory with the input profile images.

    Returns:
        dict: A dictionary with the 1D vectors
    """

    # Get the new dense landmarks from database
    updated_faces_dict = get_landmarks_from_db(DATABASE_FACE_COLLECTION, 
        'faceplpl_dense_v2', 'faceplpl_dense_msg_v2')

    # Insert new data from the 2 files
    people_input_dict = read_input_datasets()

    # Dictionary for storing all the facial vectors (e.g. landmarks vector and 
    # pixels vector) that will be used for investigating:
    #     a) emergence in entrepreneurship
    #     b) success
    data = {
        'is_entre': {
            'x_landmarks_vector': [],
            'x_pixels_vector': [],
            'y_target': [],  
            'uuids_order': [],
        },
        'rec_funding': {
            'x_landmarks_vector': [],
            'x_pixels_vector': [],
            'y_target': [],  
            'uuids_order': [],
        },
        'total_funding': {
            'x_landmarks_vector': [],
            'x_pixels_vector': [],
            'y_target': [],  
            'uuids_order': [],
        },
        'est_revenue': {
            'x_landmarks_vector': [],
            'x_pixels_vector': [],
            'y_target': [],  
            'uuids_order': [],
        },
        'recent_valuation': {
            'x_landmarks_vector': [],
            'x_pixels_vector': [],
            'y_target': [],  
            'uuids_order': [],
        },
    }

    # Find the filenames and path of the profile images stored in the local folder
    images_filenames = find_images_filenames(images_input_dir)
    images_uuid_dict = dict()
    for item_image in images_filenames:
        uuid = item_image['name'].split('.', 1)[0]

        if not uuid in updated_faces_dict:
            continue

        images_uuid_dict[uuid] = {
            'image_name': item_image['name'],
            'image_path': item_image['path'],
        }


    # Store the target (y) data for emergence in entrepreneurship and success
    # into 2 vectors
    points_dict = dict()
    for uuid in updated_faces_dict:
        obj = updated_faces_dict[uuid]

        uuid_preprocessed = uuid.replace("-", "")
        new_obj = people_input_dict[uuid_preprocessed]
        
        # DV: Emergence in Entrepreneurship
        data['is_entre']['y_target'].append(int(obj['info']['isEntre']))
        data['is_entre']['uuids_order'].append(uuid)

        # DV: Success
        if int(obj['info']['isEntre']) == 1:

            # new_has_received_funding
            if str(people_input_dict[uuid_preprocessed]['new_has_received_funding']) != '':
                data['rec_funding']['y_target'].append(int(people_input_dict[uuid_preprocessed]['new_has_received_funding']))
                data['rec_funding']['uuids_order'].append(uuid)

            # new_total_received_funding_usd
            if str(people_input_dict[uuid_preprocessed]['new_total_received_funding_usd']) != '':
                data['total_funding']['y_target'].append(int(people_input_dict[uuid_preprocessed]['new_total_received_funding_usd']))
                data['total_funding']['uuids_order'].append(uuid)

            # processed_revenue
            if str(people_input_dict[uuid_preprocessed]['processed_revenue']) != '':
                data['est_revenue']['y_target'].append(int(people_input_dict[uuid_preprocessed]['processed_revenue']))
                data['est_revenue']['uuids_order'].append(uuid)

            # processed_most_recent_valuation
            if str(people_input_dict[uuid_preprocessed]['processed_most_recent_valuation']) != '':
                data['recent_valuation']['y_target'].append(int(people_input_dict[uuid_preprocessed]['processed_most_recent_valuation']))
                data['recent_valuation']['uuids_order'].append(uuid)

        # Convert landmarks to dictionary
        for face_area in obj['faceplpl_dense_v2']['landmark']:
            for point in obj['faceplpl_dense_v2']['landmark'][face_area]:
                point_item = obj['faceplpl_dense_v2']['landmark'][face_area][point]

                if 'radius' in point:
                    continue

                if not point in points_dict:
                    points_dict[point] = []

                # for normal landmarks (NOT radius metrics) as given by Face++    
                if type(point_item) == dict: 
                    points_dict[point].append([point_item['x'], point_item['y']])

    # Convert landmarks and pixels into 1D vectors for entrepreneurship and 
    # success datasets
    for i in range(0, len(data['is_entre']['uuids_order'])):

        uuid = data['is_entre']['uuids_order'][i]
        uuid_preprocessed = uuid.replace("-", "")

        ### Landmark Vectors ###
        row_landmarks = []


        for key in points_dict:
            items_list = points_dict[key][i]
            for j in range(0, len(items_list)):
                row_landmarks.append(items_list[j])

        data['is_entre']['x_landmarks_vector'].append(row_landmarks)
        if data['is_entre']['y_target'][i] == 1:
            # data['success']['x_landmarks_vector'].append(row_landmarks)

            # new_has_received_funding
            if str(people_input_dict[uuid_preprocessed]['new_has_received_funding']) != '':
                data['rec_funding']['x_landmarks_vector'].append(row_landmarks)
            
            # new_total_received_funding_usd
            if str(people_input_dict[uuid_preprocessed]['new_total_received_funding_usd']) != '':
                data['total_funding']['x_landmarks_vector'].append(row_landmarks)

            # processed_revenue
            if str(people_input_dict[uuid_preprocessed]['processed_revenue']) != '':
                data['est_revenue']['x_landmarks_vector'].append(row_landmarks)

            # processed_most_recent_valuation
            if str(people_input_dict[uuid_preprocessed]['processed_most_recent_valuation']) != '':
                data['recent_valuation']['x_landmarks_vector'].append(row_landmarks)

        ### Pixels Vectors  ###
        row_pixels = []
        
        # Open image in grayscale mode to have only 1 number
        img = cv2.imread(images_uuid_dict[uuid]['image_path'], 0)

        # grab the image dimensions
        h, w = img.shape[:2]

        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):
                row_pixels.append(img[y, x])

        data['is_entre']['x_pixels_vector'].append(row_pixels)
        if data['is_entre']['y_target'][i] == 1:
            # data['success']['x_pixels_vector'].append(row_pixels)

            # new_has_received_funding
            if str(people_input_dict[uuid_preprocessed]['new_has_received_funding']) != '':
                data['rec_funding']['x_pixels_vector'].append(row_pixels)
            
            # new_total_received_funding_usd
            if str(people_input_dict[uuid_preprocessed]['new_total_received_funding_usd']) != '':
                data['total_funding']['x_pixels_vector'].append(row_pixels)

            # processed_revenue
            if str(people_input_dict[uuid_preprocessed]['processed_revenue']) != '':
                data['est_revenue']['x_pixels_vector'].append(row_pixels)

            # processed_most_recent_valuation
            if str(people_input_dict[uuid_preprocessed]['processed_most_recent_valuation']) != '':
                data['recent_valuation']['x_pixels_vector'].append(row_pixels)

    # Convert lists to nparrays
    for dv_key in data:
        for arr_key in data[dv_key]:
            data[dv_key][arr_key] = np.array(data[dv_key][arr_key])

    return data
###############################################################################
def PCA_LDA(data, dv_key):
    """Apply dimensonality reduction (PCA+LDA) and calculate the 2 facial
       measures for the whole face ("Pixels Intensity Measure" and 
       "Facial Landmarks Measure")

    Args:
        data (dict): The dictionary with the 1D facial vectors and dependent variables
        dv_key (str): The dependent variable name
    Returns:
        dict: A dictionary with the calculated facial measures for each person.
    """

    print('Start: PCA_LDA')
    print('Rows:', data[dv_key]['x_landmarks_vector'].shape[0])
    print('Columns:', data[dv_key]['x_landmarks_vector'].shape[1])

    data[dv_key]['people_dict'] = dict()
    Y = data[dv_key]['y_target']

    ### Standardizing the features ###
    # standard_scaler = StandardScaler()
    X_a = StandardScaler().fit_transform(data[dv_key]['x_landmarks_vector'])
    X_b = StandardScaler().fit_transform(data[dv_key]['x_pixels_vector'])

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X_a, Y):

        # x_landmarks_vector
        X_train_a, X_test_a = X_a[train_index], X_a[test_index]
        # x_pixels_vector
        X_train_b, X_test_b = X_b[train_index], X_b[test_index]
        # Target vector
        y_train, y_test = Y[train_index], Y[test_index]
        # UUIDs vector
        uuid_train = data[dv_key]['uuids_order'][train_index]
        uuid_test = data[dv_key]['uuids_order'][test_index]

        ### PCA ###
        # PCA(n_components=.99)
        pca_a = PCA(n_components=200)
        pca_b = PCA(n_components=200)

        # pca_a.fit(X_train_a)
        # pca_b.fit(X_train_b)

        pca_X_train_a = pca_a.fit_transform(X_train_a)
        pca_X_test_a = pca_a.transform(X_test_a)
        pca_X_train_b = pca_b.fit_transform(X_train_b)
        pca_X_test_b = pca_b.transform(X_test_b)

        ### LDA ###
        # Create an LDA that will reduce the data down to 1 feature
        lda_a = LinearDiscriminantAnalysis(n_components=1)
        lda_b = LinearDiscriminantAnalysis(n_components=1)

        X_train_a = lda_a.fit_transform(pca_X_train_a, y_train)
        X_test_a = lda_a.transform(pca_X_test_a)
        X_train_b = lda_b.fit_transform(pca_X_train_b, y_train)
        X_test_b = lda_b.transform(pca_X_test_b)
        

        ### Save calculated the 2 measures into the dict ###
        for i in range(0, len(X_test_a)):
            uuid = uuid_test[i].replace("-", "")
            data[dv_key]['people_dict'][uuid] = {}
            data[dv_key]['people_dict'][uuid]['facial_landmarks_measure'] = X_test_a[i][0]

        for i in range(0, len(X_test_b)):
            uuid = uuid_test[i].replace("-", "")
            data[dv_key]['people_dict'][uuid]['pixels_intensity_measure'] = X_test_b[i][0]

    print('Finish: PCA_LDA')
    return data
###############################################################################
def calc_whole_face_measures(images_input_dir):
    """Calculate the 2 facial measures for the whole face ("Pixels Intensity Measure" and 
       "Facial Landmarks Measure") and store them into MongoDB.

    Args:
        input_images_dir (str): The directory with the input profile images.

    Returns:
        -
    """
    
    # Get and convert landmarks and pixels 1D vector
    data = pixels_and_landmarks_to_1D_array(images_input_dir)

    # Apply PCA+LDA for DV=is_entre ###
    data = PCA_LDA(data, 'is_entre')

    # Apply PCA+LDA for Success DVs ###
    data = PCA_LDA(data, 'rec_funding')
    data = PCA_LDA(data, 'total_funding')
    data = PCA_LDA(data, 'est_revenue')

    # Insert new data from the 2 files
    people_input_dict, header_entre, header_success = read_input_datasets()

    output_f1 = open('Output_Datasets/DATASET_machine_learning_ENTRE.tsv', 'w')
    header_fields_1 = header_entre.split("\t")
    header_entre += '\tfacial_landmarks_measure\tpixels_intensity_measure\t'
    output_f1.write(header_entre + '\n')

    for uuid in people_input_dict:
        for i in range(0, len(header_fields_1)):
            output_f1.write(str(people_input_dict[uuid][header_fields_1[i]]) + '\t')
        
        if uuid in data['is_entre']['people_dict']:
            output_f1.write(str(data['is_entre']['people_dict'][uuid]['facial_landmarks_measure']) + '\t')
            output_f1.write(str(data['is_entre']['people_dict'][uuid]['pixels_intensity_measure']) + '\t')
        else:
            output_f1.write('\t')
            output_f1.write('\t')

        output_f1.write('\n')
    output_f1.close()


    output_f2 = open('Output_Datasets/DATASET_machine_learning_SUCCESS_10_09_2020.tsv', 'w')
    header_fields_2 = header_success.split("\t")
    header_success += '\t'
    header_success += 'RecFund_facial_landmarks_measure\tRecFund_pixels_intensity_measure\t'
    header_success += 'TotalFunding_facial_landmarks_measure\tTotalFunding_pixels_intensity_measure\t'
    header_success += 'EstRevenue_facial_landmarks_measure\tEstRevenue_pixels_intensity_measure\t'
    header_success += 'RecValuation_facial_landmarks_measure\tRecValuation_pixels_intensity_measure\t'
    output_f2.write(header_success + '\n')

    dv_names = ['rec_funding', 'total_funding', 'est_revenue', 'recent_valuation']

    for uuid in people_input_dict:
        if str(people_input_dict[uuid]['isentre']) == '1':

            for i in range(0, len(header_fields_2)):
                if header_fields_2[i] in people_input_dict[uuid]:
                    output_f2.write(str(people_input_dict[uuid][header_fields_2[i]]) + '\t')
                else:
                    output_f2.write('\t')
            
            for i in range (0, len(dv_names)):
                if uuid in data[dv_names[i]]['people_dict']:
                    output_f2.write(str(data[dv_names[i]]['people_dict'][uuid]['facial_landmarks_measure']) + '\t')
                    output_f2.write(str(data[dv_names[i]]['people_dict'][uuid]['pixels_intensity_measure']) + '\t')
                else:
                    output_f2.write('\t')
                    output_f2.write('\t')

            output_f2.write('\n')
    output_f2.close()
###############################################################################


if __name__ == '__main__':

    # ### Retrieve facial landmarks and attributes (e.g. age, gender) from the "Detect FACE++ API" ###
    faceplpl_request(INITIAL_PROFILE_IMAGES_DIR, FACEPLPL_DETECT_API, 'faceplpl_detect_initial')

    # ### Retrieve 1000 facial landmarks from the "Dense FACE++ API" ###
    faceplpl_request(INITIAL_PROFILE_IMAGES_DIR, FACEPLPL_DENSE_FACIAL_LANDMARK_API, 'faceplpl_dense_v1')

    # ### Calculate the following facial ratios: fWHR, Cheekbone and Facial Symmetry
    facial_ratios()

    # ### Crop Images ###
    crop_images(INITIAL_PROFILE_IMAGES_DIR, CROPPED_IMAGES_DIR)

    # ### Convert all images to grayscale ###
    convert_to_grayscale_images(CROPPED_IMAGES_DIR, GRAYSCALE_IMAGES_DIR)

    # ### Resize Images based on width 100 pixels ###
    resize_images(GRAYSCALE_IMAGES_DIR, RESIZED_IMAGES_DIR, new_width=100, inter=cv2.INTER_LANCZOS4)

    # ### Pad with extra black pixels the image and bring them all to the same height ###
    pad_images(RESIZED_IMAGES_DIR, PADDED_IMAGES_DIR)

    # ### Retrieve the updated 1000 facial landmarks from the "Dense FACE++ API" ###
    faceplpl_request(PADDED_IMAGES_DIR, FACEPLPL_DENSE_FACIAL_LANDMARK_API, 'faceplpl_dense_v2')

    # ### Remove background using face++ points ###
    remove_background(PADDED_IMAGES_DIR, BACKGROUND_IMAGES_DIR)

    # ### Use Histogram Equalization to improve contrast in images ###
    histogram_equalization(BACKGROUND_IMAGES_DIR, EQ_HIST_IMAGES_DIR)

    ### Calculate the measures "Pixels Intensity Measure" and "Facial Landmarks Measure" ###
    calc_whole_face_measures(EQ_HIST_IMAGES_DIR)
    