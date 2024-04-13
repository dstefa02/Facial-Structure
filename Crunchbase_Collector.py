# encoding: utf-8

"""Download Crunchbase data using the Crunchbase API v3.1 
   (https://data.crunchbase.com/v3.1/reference) 
   and store them to a MongoDB.
"""

###############################################################################
# The following code is distributed under MIT license. Details are withheld
# for purposes of anonymity.
###############################################################################

###############################################################################
### Libraries ###
from gevent import monkey
monkey.patch_all()

import pymongo
from pymongo import MongoClient
import os, sys, datetime, re, time, json, traceback
from sys import stdout
import grequests, requests
from os import path
from io import BytesIO
from PIL import Image
###############################################################################

###############################################################################
### Global Variables ###
# Key for accessing the Crunchbase API
CRUNCHBASE_USER_KEY = "INSERT_CRUNCHBASE_KEY"
# Max requests per minute (defined by Crunchbase API)
CRUNCHBASE_MAX_REQUESTS_PER_MINUTE = 200
# Max returned items per page (defined by Crunchbase API)
CRUNCHBASE_MAX_ITEMS_PER_PAGE = "250"
# The number of parallel requests using grequests
NUM_PARALLEL_REQUESTS = 10

# Crunchbase API Endpoint "Collections" - Collection endpoints to retrieve the 
# entire set and core properties of many of the important Item types.
CRUNCHBASE_COLLECTIONS_ENDPOINT = (
    "https://api.crunchbase.com/v3.1/<:COLLECTION_NAME:>"
    "?items_per_page=<:ITEMS_PER_PAGE:>&user_key=<:USER_KEY:>")
# Crunchbase API Endpoint "Item Details" - Item Detail endpoint to retrieve 
# not only the core properties of each Node but also the details of related Items.
CRUNCHBASE_ITEM_DETAILS_ENDPOINT = (
    "https://api.crunchbase.com/v3.1/<:COLLECTION_NAME:>/"
    "<:ITEM_UUID:>?user_key=<:USER_KEY:>")
# Crunchbase API Endpoint "Refined Searches" - When you query an entity, the 
# set of connected Nodes by RELATIONSHIP_NAME can be retrieved. 
CRUNCHBASE_ITEM_RELAT_ENDPOINT = (
    "https://api.crunchbase.com/v3.1/<:COLLECTION_NAME:>/"
    "<:ITEM_UUID:>/<:RELATIONSHIP_NAME:>"
    "?items_per_page=<:ITEMS_PER_PAGE:>&user_key=<:USER_KEY:>")

# Database ip
DATABASE_IP = "localhost"
# Database port
DATABASE_PORT = 27017
# Database name for storing Crunchbase data
DATABASE_NAME = "Crunchbase"
# Dictionary with stored connections with database (MongoDB)
db_connections = {}

# Log file for storing errors
error_log_file = open("LOG-ERROR_Crunchbase_Collector.log", "w")
# Log file for storing the progress
progress_log_file = open("LOG-PROGRESS_Crunchbase_Collector.log", "w")

# Directory to store profile images
PROFILE_IMAGES_DIR = "Crunchbase_Profile_Images"

###############################################################################
def test_crunchbase_api():
    """Test the Crunchbase API and the validity of the Crunchbase Key.

    Args:
        -

    Returns:
        -
    """

    # Create test URL
    URL = CRUNCHBASE_COLLECTIONS_ENDPOINT.replace("<:COLLECTION_NAME:>", 'people')
    URL = URL.replace("<:ITEMS_PER_PAGE:>", CRUNCHBASE_MAX_ITEMS_PER_PAGE)
    URL = URL.replace("<:USER_KEY:>", CRUNCHBASE_USER_KEY)

    # Test a request
    response = requests.request("GET", URL)
    try:
        max_pages = response.json()['data']['paging']['number_of_pages']
    except: # Unauthorized user_key
        print("\n", "Error: ", response.json())
        error_log_file.write("\nError: " + str(response.json()) + "\n")
        error_log_file.write(str(traceback.format_exc()))
        error_log_file.write("-"*50 + "\n")
        exit()
###############################################################################
def download_crunchbase_main_collections(collection_name):
    """Retrieve the main info of a specific collection from Crunchbase API

    Args:
        collection_name (str): The name of the collection/Node to be collected from Crunchbase

    Returns:
        -
    """
    # Create URL for this crunchbase collection
    URL = CRUNCHBASE_COLLECTIONS_ENDPOINT
    URL = URL.replace("<:COLLECTION_NAME:>", collection_name.replace("_", "-"))
    URL = URL.replace("<:ITEMS_PER_PAGE:>", CRUNCHBASE_MAX_ITEMS_PER_PAGE)
    URL = URL.replace("<:USER_KEY:>", CRUNCHBASE_USER_KEY)

    # Get max pages of this collection
    response = requests.request("GET", URL)
    max_pages = response.json()['data']['paging']['number_of_pages']

    time.sleep(1)

    # Connect to MongoDB
    client = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections[collection_name] = client[DATABASE_NAME][collection_name]

    # Download collection
    progress_log_file.write("\nStart download_crunchbase_main_collections " 
        + collection_name +"...\n")
    progress_log_file.write("="*50 + "\n")
    start_time = time.time()
    count_requests_per_minute, page = 0, 1
    urls = []

    while page <= max_pages: # download all nodes per page
        urls.append(URL + "&page=" + str(page)) 

        if len(urls) == NUM_PARALLEL_REQUESTS:
            parallel_get_requests_main_collections(urls, collection_name)

            urls = []
            count_requests_per_minute += NUM_PARALLEL_REQUESTS
            
            progress_log_file.write("Page: " + str(page) + ", " + "Max page: " + 
                str(max_pages) + ", " + "Time: " + 
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            error_log_file.flush()
            progress_log_file.flush()
        page += 1

    # If there are still some urls/pages to retrieve
    if len(urls) > 0: 
        parallel_get_requests_main_collections(urls, collection_name)
        progress_log_file.write("Page: " + str(page) + ", " + "Max page: " + 
            str(max_pages) + ", " + "Time: " + 
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    progress_log_file.write("\nFinished download_crunchbase_main_collections " + 
        collection_name +".\n" + "="*50 + "\n")
###############################################################################
def parallel_get_requests_main_collections(urls, collection_name):
    """Send multiple parallel requests to Crunchbase API and store the results 
       to the database.

    Args:
        urls (list): The list of the URLs to be requested
        collection_name (str): The name of the collection/Node to be collected 
        from Crunchbase

    Returns:
        -
    """

    total_urls = len(urls)

    # Send requests and process results
    for i in range(0, total_urls):
        res = requests.request("GET", urls[i])

        try:
            res_json = res.json()

            for obj in res_json['data']['items']:
                obj['_id'] = obj['uuid']
                res.close()

                # Store items to mongo
                try:
                    db_connections[collection_name].insert(obj)
                except pymongo.errors.DuplicateKeyError as e:
                    pass

        except:
            try:
                # If Usage limit exceeded then wait for before sending a new request
                if str(res.text) == "Usage limit exceeded":
                    progress_log_file.write("="*50 + "\nSleep for " + str(30) + 
                        " seconds...\n" + "="*50 + "\n")
                    progress_log_file.flush()
                    time.sleep(30)
                    # Request the remaining urls
                    parallel_get_requests_main_collections(urls[i:], collection_name)
                else:
                    err = str(traceback.format_exc())
                    error_log_file.write("-"*50 + "\n")
                    error_log_file.write("\nERROR: \n" + err + "\n\nPages: " + urls[i] + 
                        "\nRes: " + str(res.text) + "\n")
                    error_log_file.flush()
            except:
                err = str(traceback.format_exc())
                error_log_file.write("-"*50 + "\n")
                error_log_file.write("\nERROR: \n" + err + "\n\nPages: " + urls[i] + "\n")
                error_log_file.write("\nCollection: " + collection_name + "\n")
                error_log_file.flush()
###############################################################################
def download_crunchbase_item_details(collection_name):
    """Retrieve the details of items from a specific collection from Crunchbase API

    Args:
        collection_name (str): The name of the collection/Node to be collected from Crunchbase

    Returns:
        -
    """

    # Connect to MongoDB
    client_1 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections[collection_name] = client_1[DATABASE_NAME][collection_name]

    client_2 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections['Relationships_'+collection_name] = client_2[DATABASE_NAME]['Relationships_'+collection_name]

    # Get collection from MongoDB
    collection_list = []
    for obj in db_connections[collection_name].find({"all_properties": {"$exists": 0}},{"uuid": 1}):
        collection_list.append(obj)

    # Download details of items
    progress_log_file.write("\nStart download_crunchbase_item_details "+ collection_name +" ...\n")
    progress_log_file.write("="*50 + "\n")
    start_time = time.time()
    count_requests_per_minute, progress_count = 0, 0
    urls, collection_obj = [], []

    for obj in collection_list:

        # Create URL for this crunchbase collection
        URL = CRUNCHBASE_ITEM_DETAILS_ENDPOINT
        URL = URL.replace("<:COLLECTION_NAME:>", collection_name.replace("_", "-"))
        URL = URL.replace("<:ITEM_UUID:>", obj['uuid'])
        URL = URL.replace("<:USER_KEY:>", CRUNCHBASE_USER_KEY)

        urls.append(URL)
        collection_obj.append(obj)

        if len(urls) == NUM_PARALLEL_REQUESTS:
            parallel_get_requests_item_details(urls, collection_name, collection_obj)
            urls, collection_obj = [], []
            count_requests_per_minute += NUM_PARALLEL_REQUESTS
            
            progress_log_file.write("Counter: " + str(progress_count) + ", " + 
                "Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
            error_log_file.flush()
            progress_log_file.flush()   

        progress_count += 1

    if len(urls) > 0:
        parallel_get_requests_item_details(urls, collection_name, collection_obj)
        progress_log_file.write("Counter: " + str(progress_count) + ", " + 
            "Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    progress_log_file.write("\nFinished download_crunchbase_item_details "+ 
        collection_name +".\n" + "="*50 + "\n")
###############################################################################
def parallel_get_requests_item_details(urls, collection_name, collection_obj):
    """Send multiple parallel requests to Crunchbase API and store the results 
       to the database.

    Args:
        urls (list): The list of the URLs to be requested
        collection_name (str): The name of the collection/Node to be collected 
                               from Crunchbase
        collection_obj (str): The uuids of the items to be collected

    Returns:
        -
    """

    total_urls = len(urls)
    # Send requests and process results
    rs = (grequests.get(url, allow_redirects=False, timeout=10) for url in urls)
    resp_list = grequests.map(rs)

    for i in range(0, total_urls):
        cur_obj = collection_obj[i]
        res = resp_list[i]
        # res = requests.request("GET", urls[i])

        try:
            res_json = res.json()

            relationships = res_json['data']['relationships']
            for relat_name in relationships:

                if not relat_name in db_connections:
                    # Connect to MongoDB
                    client = MongoClient(DATABASE_IP, DATABASE_PORT)
                    db_connections[relat_name] = client[DATABASE_NAME][relat_name]

                if 'item' in relationships[relat_name]:
                    relationships[relat_name]['item']['_id'] = relationships[relat_name]['item']['uuid']
                    
                    try:
                        # insert to secondary collection
                        db_connections[relat_name].insert(relationships[relat_name]['item'])
                    except pymongo.errors.DuplicateKeyError as e:
                        pass

                    try:
                        # insert relationship
                        db_connections['Relationships_'+collection_name].insert({
                            '_id': { 
                                'primary_uuid': res_json['data']['uuid'],
                                'secondary_uuid': relationships[relat_name]['item']['uuid'], 
                                'relationship_name': relat_name
                            }
                        })
                    except pymongo.errors.DuplicateKeyError as e:
                        pass
                    
                    relationships[relat_name].pop("item", None)
                elif 'items' in relationships[relat_name]:
                    for item in relationships[relat_name]['items']:
                        item['_id'] = item['uuid']

                        try:
                            # insert to secondary collection
                            db_connections[relat_name].insert(item)
                        except pymongo.errors.DuplicateKeyError as e:
                            pass

                        try:
                            # insert relationship
                            db_connections['Relationships_'+collection_name].insert({
                                '_id': { 
                                    'primary_uuid': res_json['data']['uuid'],
                                    'secondary_uuid': item['uuid'], 
                                    'relationship_name': relat_name
                                }
                            })
                        except pymongo.errors.DuplicateKeyError as e:
                            pass

                    relationships[relat_name].pop("items", None)

            # update main collection
            db_connections[collection_name].update_one(
                {'_id': res_json['data']['uuid']},
                {'$set': {
                        'all_properties': res_json['data']['properties'], 
                        'relationships': relationships,
                        'relationships_status': 0
                    }
                }
            )

        except Exception as e:

            try:
                # If Usage limit exceeded then wait for 60 seconds before sending a new request
                if str(res.text) == "Usage limit exceeded":
                    progress_log_file.write("="*50 + "\nSleep for " + str(15)  + " seconds...\n" + "="*50 + "\n")
                    progress_log_file.flush()
                    time.sleep(15)

                    # Request the remaining urls
                    parallel_get_requests_item_details(urls[i:], collection_name, collection_obj[i:])
                else:
                    err = str(traceback.format_exc())
                    error_log_file.write("-"*50)
                    error_log_file.write("\nERROR: \n" + err + "\n\nUrl: " + urls[i] + "\nRes: " + str(res.text) + "\n")
                    error_log_file.write("\nCollection: " + collection_name + "\n")
                    error_log_file.flush()
            except:
                err = str(traceback.format_exc())
                error_log_file.write("-"*50 + "\n")
                error_log_file.write("\nERROR: \n" + err + "\n\nPages: " + urls[i] + "\n")
                error_log_file.write("\nCollection: " + collection_name + "\n")
                error_log_file.flush()
###############################################################################
def download_crunchbase_item_relationships(collection_name):
    """Retrieve the relationships of items from a specific collection from Crunchbase API

    Args:
        collection_name (str): The name of the collection/Node that its relationships will be collected from Crunchbase

    Returns:
        -
    """

    # Download collection
    progress_log_file.write("\nStart download_crunchbase_item_relationships "+ collection_name +"...\n")
    progress_log_file.write("="*50 + "\n")
    progress_log_file.flush()
    count_progress, count_requests_per_minute = 0, 0
    start_time = time.time()

    # Connect to db
    client_1 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections[collection_name] = client_1[DATABASE_NAME][collection_name]
    client_2 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections['Relationships_' + collection_name] = client_2[DATABASE_NAME]['Relationships_' + collection_name]

    # Get collection from MongoDB
    query = {'$and': [{'relationships': {'$exists': 1}}, {'relationships_status': 0}]}
    project = {'uuid': 1, 'properties.permalink':1, 'relationships': 1, 'relationships_status': 1}
    # collection_list = []
    for obj in db_connections[collection_name].find(query, project, no_cursor_timeout=True):
        for relat_name in obj['relationships']:
            
            if 'items' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('items', None)

            if 'item' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('item', None)

            if 'cardinality' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('cardinality', None)

            if not 'paging' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name]['paging'] = {}

            if not 'total_items' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging']['total_items'] = 99999999

            if 'sort_order' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging'].pop('sort_order', None)

            if 'first_page_url' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging'].pop('first_page_url', None)
            
            #     collection_list.append(obj)
            # exit()
            # for obj in collection_list:
            # for relat_name in obj['relationships']:

            if obj['relationships'][relat_name]['paging']['total_items'] > 10:
                
                total_items = obj['relationships'][relat_name]['paging']['total_items']
                progress_log_file.write("Counter: " + str(count_progress) + ", " + "UUID: " + str(obj['_id']) + " , " +
                    "Relat_name: " + str(relat_name) + ", " +  "Total_items: " + str(total_items) + ", " + 
                    "Time: " + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "\n"
                )
                progress_log_file.flush()


                # Create URL for request
                URL = CRUNCHBASE_ITEM_RELAT_ENDPOINT.replace("<:COLLECTION_NAME:>", collection_name.replace("_", "-"))
                URL = URL.replace("<:ITEM_UUID:>", obj['uuid'])
                URL = URL.replace("<:RELATIONSHIP_NAME:>", relat_name)
                URL = URL.replace("<:ITEMS_PER_PAGE:>", CRUNCHBASE_MAX_ITEMS_PER_PAGE)
                URL = URL.replace("<:USER_KEY:>", CRUNCHBASE_USER_KEY)

                start_time, count_requests_per_minute = get_item_relationship(URL, obj['uuid'], collection_name, relat_name, start_time, count_requests_per_minute)

        # update main collection that relationships are finished 
        db_connections[collection_name].update_one(
            {'_id': obj['_id']},
            {'$set': {
                    'relationships_status': 1
                }
            }
        )

        count_progress += 1

    progress_log_file.write("\nFinished download_crunchbase_item_relationships "+ collection_name +".\n" + "="*50 + "\n")
###############################################################################
def download_crunchbase_item_relationships_v2(collection_name):
    """Retrieve the relationships of items from a specific collection from Crunchbase API

    Args:
        collection_name (str): The name of the collection/Node that its relationships will be collected from Crunchbase

    Returns:
        -
    """

    # Download collection
    progress_log_file.write("\nStart download_crunchbase_item_relationships "+ collection_name +"...\n")
    progress_log_file.write("="*50 + "\n")
    progress_log_file.flush()
    count_progress, count_requests_per_minute = 0, 0
    start_time = time.time()

    # Connect to db
    client_1 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections[collection_name] = client_1[DATABASE_NAME][collection_name]
    client_2 = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections['Relationships_' + collection_name] = client_2[DATABASE_NAME]['Relationships_' + collection_name]

    # Get collection from MongoDB
    query = {'$and': [{'relationships': {'$exists': 1}}, {'relationships_status': 0}]}
    project = {
        'uuid': 1, 'properties.permalink':1, 'relationships': 1, 'relationships_status': 1,
        'properties.twitter_url':1
    }
    
    tmp_file = open('tmp_file.txt', 'w')
    total_objects = 0

    for obj in db_connections[collection_name].find(query, project, no_cursor_timeout=True):
        for relat_name in obj['relationships']:
            
            if 'items' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('items', None)

            if 'item' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('item', None)

            if 'cardinality' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name].pop('cardinality', None)

            if not 'paging' in obj['relationships'][relat_name]:
                obj['relationships'][relat_name]['paging'] = {}

            if not 'total_items' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging']['total_items'] = 99999999

            if 'sort_order' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging'].pop('sort_order', None)

            if 'first_page_url' in obj['relationships'][relat_name]['paging']:
                obj['relationships'][relat_name]['paging'].pop('first_page_url', None)
        
        
        # if obj['properties']['twitter_url'] == None:
        #     continue
        
        tmp_file.write(json.dumps(obj) + '\n')
        total_objects += 1

    tmp_file.close()

    tmp_file = open('tmp_file.txt', 'r')

    for line in tmp_file:
        obj = json.loads(line.strip('\n'))

        for relat_name in obj['relationships']:

            if obj['relationships'][relat_name]['paging']['total_items'] > 10:
                
                total_items = obj['relationships'][relat_name]['paging']['total_items']
                progress_log_file.write(
                    "Counter: " + str(count_progress) + "/" +  str(total_objects) + " , " +
                    "UUID: " + str(obj['_id']) + " , " +
                    "Relat_name: " + str(relat_name) + ", " +  "Total_items: " + str(total_items) + ", " + 
                    "Time: " + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "\n"
                )
                progress_log_file.flush()


                # Create URL for request
                URL = CRUNCHBASE_ITEM_RELAT_ENDPOINT.replace("<:COLLECTION_NAME:>", collection_name.replace("_", "-"))
                URL = URL.replace("<:ITEM_UUID:>", obj['uuid'])
                URL = URL.replace("<:RELATIONSHIP_NAME:>", relat_name)
                URL = URL.replace("<:ITEMS_PER_PAGE:>", CRUNCHBASE_MAX_ITEMS_PER_PAGE)
                URL = URL.replace("<:USER_KEY:>", CRUNCHBASE_USER_KEY)

                start_time, count_requests_per_minute = get_item_relationship(URL, obj['uuid'], collection_name, relat_name, start_time, count_requests_per_minute)

        # update main collection that relationships are finished 
        db_connections[collection_name].update_one(
            {'_id': obj['_id']},
            {'$set': {
                    'relationships_status': 1
                }
            }
        )

        count_progress += 1

    progress_log_file.write("\nFinished download_crunchbase_item_relationships "+ collection_name +".\n" + "="*50 + "\n")
###############################################################################
def get_item_relationship(main_URL, main_obj_uuid, collection_name, relat_name, 
    start_time, count_requests_per_minute):
    """Send multiple parallel requests to Crunchbase API and store the results 
       to the database.

    Args:
        main_URL (str): The basic URL for a specific relationship of an item
        main_obj_uuid (str): The uuid/id of the item/Node that its relationships 
             are collected
        collection_name (str): The name of the collection/Node to be collected 
           from Crunchbase
        relat_name (str): The name of the relationship to be collected from Crunchbase
        start_time (float): A variable that monitors time per N requests
        count_requests_per_minute (int): A counter for the requests that have been done 
            till now 

    Returns:
        int: The updated counter for the requests
        float: The updated variable that monitors time per N requests
    """

    try:
        # Get number of pages
        res = requests.get(main_URL, allow_redirects=False)
        time.sleep(1)
        res_json = res.json()
        num_of_pages = res_json['data']['paging']['number_of_pages']
    except:
        error_log_file.write("-"*30)
        error_log_file.write("\n" + "ERROR: \n" + str(traceback.format_exc()) + "\n" + main_URL + "\n")
        # error_log_file.write(str(res) + "\n")
        error_log_file.write("\nCollection: " + collection_name + "\n")
        error_log_file.flush()       
        return start_time, count_requests_per_minute

    page = 1
    urls = []

    while page <= num_of_pages:
        urls.append(main_URL + "&page=" + str(page))

        if len(urls) == NUM_PARALLEL_REQUESTS:
            parallel_get_requests_item_relationships(main_obj_uuid, collection_name, relat_name, urls)
            urls = []
            count_requests_per_minute += NUM_PARALLEL_REQUESTS
            
            progress_log_file.write("      " + "Last_Page: " + str(page) + ", " + 
                "Time: " + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "\n")

            error_log_file.flush()
            progress_log_file.flush()
        page += 1

    if len(urls) > 0:
        parallel_get_requests_item_relationships(main_obj_uuid, collection_name, relat_name, urls)
        progress_log_file.write("      " + "Last_Page: " + str(page-1) + ", " + 
            "Time: " + datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "\n")

    return start_time, count_requests_per_minute
###############################################################################
def parallel_get_requests_item_relationships(main_obj_uuid, collection_name, 
    relat_name, urls):
    """Send multiple parallel requests to Crunchbase API and store the results 
       to the database.

    Args:
        main_obj_uuid (str): The uuid/id of the item/Node that its relationships 
            are collected
        collection_name (str): The name of the collection/Node to be collected 
            from Crunchbase
        relat_name (str): The name of the relationship to be collected from 
            Crunchbase
        urls (list): The list of the URLs to be requested

    Returns:
        -
    """

    total_urls = len(urls)
    # Send requests and process results
    rs = (grequests.get(url, allow_redirects=False, timeout=10) for url in urls)
    resp_list = grequests.map(rs)

    for i in range(0, total_urls):
        res = resp_list[i]
        # res = requests.request("GET", urls[i])
    
        try:
            res_json = res.json()
            data = res_json['data']

            if not relat_name in db_connections:
                # Connect to MongoDB
                client = MongoClient(DATABASE_IP, DATABASE_PORT)
                db_connections[relat_name] = client[DATABASE_NAME][relat_name]

            if 'item' in data:
                data['item']['_id'] = data['item']['uuid']

                try:
                    # insert to secondary collection
                    db_connections[relat_name].insert(data['item'])
                except pymongo.errors.DuplicateKeyError as e:
                    pass

                try:
                    # insert relationship
                    db_connections['Relationships_'+collection_name].insert({
                        '_id': { 
                              'primary_uuid': main_obj_uuid,
                              'secondary_uuid': data['item']['uuid'], 
                              'relationship_name': relat_name
                        }
                    })
                except pymongo.errors.DuplicateKeyError as e:
                    pass

            elif 'items' in data:

                for item in data['items']:
                    item['_id'] = item['uuid']

                    try:
                        # insert to secondary collection
                        db_connections[relat_name].insert(item)
                    except pymongo.errors.DuplicateKeyError as e:
                        pass

                    try:
                        # insert relationship
                        db_connections['Relationships_'+collection_name].insert({
                            '_id': {
                                  'primary_uuid': main_obj_uuid,
                                  'secondary_uuid': item['uuid'], 
                                  'relationship_name': relat_name
                            }
                        })
                    except pymongo.errors.DuplicateKeyError as e:
                        pass

        except Exception as e:
            try:
                # If Usage limit exceeded then wait for 60 seconds before sending a new request
                if str(res.text) == "Usage limit exceeded":
                    progress_log_file.write("="*50 + "\nSleep for " + str(15)  + 
                        " seconds...\n" + "="*50 + "\n")
                    progress_log_file.flush()
                    time.sleep(15)
                    # Request the remaining urls
                    parallel_get_requests_item_relationships(main_obj_uuid, collection_name, relat_name, urls[i:0])
                else:
                    err = str(traceback.format_exc())
                    error_log_file.write("-"*50 + "\n")
                    error_log_file.write("\nERROR: \n" + err + "\n\nUrl: " + urls[i] + 
                        "\nRes: " + str(res.text) + "\n")
                    error_log_file.write("\nCollection: " + collection_name + "\n")
                    error_log_file.flush()
            except:
                err = str(traceback.format_exc())
                error_log_file.write("-"*50 + "\n")
                error_log_file.write("\nERROR: \n" + err + "\n\nPages: " + urls[i] + "\n")
                error_log_file.write("\nCollection: " + collection_name + "\n")
                error_log_file.flush()
###############################################################################
def download_people_images(collection_name):
    """Retrieve the profile images of all founders from Crunchbase API

    Args:
        collection_name (str): The name of the collection/Node that its profile 
            photos will be collected from Crunchbase

    Returns:
        -
    """

    # Check if directory for storing images does not exist and create the directory
    if not os.path.exists(PROFILE_IMAGES_DIR):
        os.mkdir(PROFILE_IMAGES_DIR)

    # Connect to MongoDB
    client = MongoClient(DATABASE_IP, DATABASE_PORT)
    db_connections[collection_name] = client[DATABASE_NAME][collection_name]

    # Get people profile photos urls from MongoDB
    people_list = {}
    for obj in db_connections[collection_name].find({}, {"uuid": 1, "properties.profile_image_url":1}):
        if obj['properties']['profile_image_url'] != None:
            people_list[obj['_id']] = obj['properties']['profile_image_url']

    progress_log_file.write("\nStart download_people_images...\n" + "="*50 + "\n")

    i = 0
    for person_uuid in people_list:
        try:
            filename, ext = os.path.splitext(people_list[person_uuid])
            ext = ext.replace('.','')

            res = requests.get(people_list[person_uuid], allow_redirects=True)
            photo = res.content
            ext_2 = res.headers.get('content-type').replace("image/", "")

            im = Image.open(BytesIO(photo))
            im.save('test.' + str(ext_2))

            image_size = round(os.path.getsize("test." + str(ext_2))/1024.0/1024.0,2)

            if image_size > 2:
                size_tuple = (int(im.size[0]/2), int(im.size[1]/2))
                new_image = im.resize(size_tuple)
                photo = new_image.tobytes()

            filename = str(PROFILE_IMAGES_DIR + "/" + person_uuid + "." + ext_2)
            outfile = open(filename,'wb')
            outfile.write(photo)
            outfile.close()

            progress_log_file.write(str(i) + ": New Image Downloaded Successfully" + "\n")
            progress_log_file.flush()
        except Exception as e:
            err = str(traceback.format_exc())    
            error_log_file.write("="*50 + "\n")
            error_log_file.write(str(i) + ": New Image Failed" + "\nERROR: \n" + err + "\n")
            error_log_file.write("\nCollection: " + collection_name + "\n")  
            error_log_file.flush()  
        i += 1    

    progress_log_file.write("\nFinished download_people_images\n" + "="*50 + "\n")
###############################################################################


if __name__ == "__main__":

    progress_log_file.write("\nStart Collecting: " + 
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +".\n" + "="*50 + "\n")

    # Test Crunchbase API and check Crunchbase key
    test_crunchbase_api()

    # # Retrieve the main info of a specific collection from Crunchbase
    download_crunchbase_main_collections("people")
    download_crunchbase_main_collections("organizations")
    download_crunchbase_main_collections("funding_rounds")
    download_crunchbase_main_collections("categories")

    # Retrieve the details of items from a specific collection from Crunchbase
    download_crunchbase_item_details("people")
    download_crunchbase_item_details("organizations")
    download_crunchbase_item_details("funding_rounds")

    # Retrieve the relationships of items from a specific collection from Crunchbase
    download_crunchbase_item_relationships_v2("people")
    download_crunchbase_item_relationships_v2("organizations")
    download_crunchbase_item_relationships_v2("funding_rounds")

    # Retrieve the profile images of all people
    download_people_images("founders")
    
    progress_log_file.write("\nFinish Collecting: " + 
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +".\n" + "="*50 + "\n")