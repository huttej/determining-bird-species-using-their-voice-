from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import os
#Import libraries
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = './cnn_50.h5'  # Ensure this is the correct path to your saved model
model = load_model(model_path)

# labels = ['Acridotherestristis', 'Aegithaloscaudatus', 'Alaudaarvensis',
#  'Andean Guan_sound', 'Andean Tinamou_sound', 'Apusapus',
#  'Band-tailed Guan_sound', 'Cacicuscela', 'Cardueliscarduelis',
#  'Cauca Guan_sound', 'Chlorischloris', 'Coccothraustescoccothraustes',
#  'Columbalivia', 'Columbapalumbus', 'Corvuscorone', 'Corvusfrugilegus',
#  'Cuculuscanorus', 'Delichonurbicum', 'Dendrocoposmajor',
#  'Dumetellacarolinensis', 'East Brazilian Chachalaca_sound',
#  'Emberizacitrinella', 'Erithacusrubecula', 'Ficedulahypoleuca',
#  'Fringillacoelebs', 'Gallusgallus', 'Garrulusglandarius', 'Hirundorustica',
#  'Laniusexcubitor', 'Luscinialuscinia', 'Motacillaalba', 'Motacillaflava',
#  'Parusmajor', 'Passerdomesticus', 'Phoenicurusochruros',
#  'Phoenicurusphoenicurus', 'Phylloscopuscollybita', 'Phylloscopustrochilus',
#  'Picapica', 'Pycnonotuscafer', 'Pycnonotusjocosus', 'Sittaeuropaea',
#  'Streptopeliaturtur', 'Sturnusvulgaris', 'Troglodytestroglodytes',
#  'Turdusmerula', 'Turdusphilomelos', 'Turduspilaris', 'Upupaepops',
#  'Variegated Tinamou_sound'
# ]

labels = [
    'Acridotheres tristis',   # Common Myna
    'Aegithalos caudatus',     # Long-tailed Tit
    'Alauda arvensis',         # Skylark
    'Andean Guan', 'Andean Tinamou',
    'Apus apus',               # Common Swift
    'Band-tailed Guan',
    'Cacicus cela',            # Yellow-shouldered Blackbird
    'Carduelis carduelis',     # European Goldfinch
    'Cauca Guan',
    'Chloris chloris',         # European Greenfinch
    'Coccothraustes coccothraustes',  # Hawfinch
    'Columba livia',           # Rock Pigeon
    'Columba palumbus',        # Common Wood Pigeon
    'Corvus corone',           # Carrion Crow
    'Corvus frugilegus',       # Rook
    'Cuculus canorus',         # Common Cuckoo
    'Delichon urbicum',        # House Martin
    'Dendrocopos major',       # Great Spotted Woodpecker
    'Dumetella carolinensis',  # Eastern Towhee
    'East Brazilian Chachalaca',
    'Emberiza citrinella',     # Yellowhammer
    'Erithacus rubecula',      # European Robin
    'Ficedula hypoleuca',      # European Pied Flycatcher
    'Fringilla coelebs',       # Common Chaffinch
    'Gallus gallus',          # Red Junglefowl
    'Garrulus glandarius',      # Eurasian Jay
    'Hirundo rustica',         # Barn Swallow
    'Lanius excubitor',        # Northern Shrike
    'Luscinia luscinia',       # Nightingale
    'Motacilla alba',          # White Wagtail
    'Motacilla flava',         # Yellow Wagtail
    'Parus major',             # Great Tit
    'Passer domesticus',       # House Sparrow
    'Phoenicurus ochruros',    # Black Redstart
    'Phoenicurus phoenicurus',  # Common Redstart
    'Phylloscopus collybita',  # Common Chiffchaff
    'Phylloscopus trochilus',  # Willow Warbler
    'Pica pica',               # Eurasian Magpie
    'Pycnonotus cafer',        # Black-crowned Night Heron
    'Pycnonotus jocosus',      # Eurasian Blackbird
    'Sitta europaea',          # Eurasian Nuthatch
    'Streptopelia turtur',     # European Turtle Dove
    'Sturnus vulgaris',        # Common Starling
    'Troglodytes troglodytes',  # Eurasian Wren
    'Turdus merula',           # Common Blackbird
    'Turdus philomelos',       # Song Thrush
    'Turdus pilaris',          # Fieldfare
    'Upupa epops',             # Eurasian Hoopoe
    'Variegated Tinamou'
]

# Feature extraction functions
def add_noise(data, random=False, rate=0.035, threshold=0.075):
    if random:
        rate = np.random.random() * threshold
    noise = rate * np.random.uniform() * np.amax(data)
    augmented_data = data + noise * np.random.normal(size=data.shape[0])
    return augmented_data

def pitching(data, sr, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_features(path, duration=2.5, offset=0.6):
    # Load audio file
    data, sr = librosa.load(path, duration=duration, offset=offset)
    
    # Extract features
    aud = extract_features(data, sr)
    audio = np.array(aud)
    
    # Apply data augmentation techniques and extract features
    noised_audio = add_noise(data, random=True)
    aud2 = extract_features(noised_audio, sr)
    audio = np.vstack((audio, aud2))
    
    pitched_audio = pitching(data, sr, random=True)
    aud3 = extract_features(pitched_audio, sr)
    audio = np.vstack((audio, aud3))
    
    return audio

# Preprocess single audio input
def preprocess_audio(audio_path, scaler):
    # Extract features
    features = get_features(audio_path)
    
    # Scale the features using the same scaler used during training
    scaled_features = scaler.transform(features)
    
    # Expand dimensions to match the input shape of the CNN model
    scaled_features = np.expand_dims(scaled_features, axis=2)
    
    return scaled_features
# Load the scaler used during training (assuming it's saved as 'scaler.pkl')
import joblib
scaler = joblib.load('./scaler_50.pkl') 

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# In-memory user storage (replaces MySQL database)
users_db = []


@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/About')
def about():
    return render_template('about.html')

# Registration and login removed - skipping directly to upload
# Users can now use the upload page without authentication


@app.route('/home')
def home():
    return render_template('home.html')


# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         if 'audio' not in request.files:
#             return render_template("upload.html", message="No file part")

#         myfile = request.files['audio']
        
#         if myfile.filename == '':
#             return render_template("upload.html", message="No selected file")

#         accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
#         if not myfile.filename.split('.')[-1].lower() in accepted_formats:
#             message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
#             return render_template("upload.html", message=message)
        
#         filename = myfile.filename
#         mypath = os.path.join('static/audio/', filename)
#         myfile.save(mypath)
#         # Preprocess the audio input
#         processed_audio = preprocess_audio(mypath, scaler)
#         # Predict the class
#         prediction = model.predict(processed_audio)
#         predicted_class = np.argmax(prediction, axis=1)

#         # Map predicted class index back to label
#         predicted_bird = labels[predicted_class[0]]
#         print(f'Predicted Bird Name: {predicted_bird}')
#         return render_template('upload.html', prediction=predicted_bird, path=mypath, recommendations=recommendations)
    
#     return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the audio file is part of the request
        if 'audio' not in request.files:
            return render_template("upload.html", message="No file part")
        
        myfile = request.files['audio']
        
        # Check if a file is selected
        if myfile.filename == '':
            return render_template("upload.html", message="No selected file")

        # Validate file format
        accepted_formats = ['mp3', 'wav', 'ogg', 'flac']
        if not myfile.filename.split('.')[-1].lower() in accepted_formats:
            message = "Invalid file format. Accepted formats: {}".format(', '.join(accepted_formats))
            return render_template("upload.html", message=message)
        
        # Save the audio file
        filename = myfile.filename
        mypath = os.path.join('static/audio/', filename)
        myfile.save(mypath)
        
        # Preprocess the audio input
        processed_audio = preprocess_audio(mypath, scaler)
        
        # Predict the class
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction, axis=1)
        print("Prediction output:", prediction)

        # Map predicted class index back to label
        predicted_bird = labels[predicted_class[0]]
        print(f'Predicted Bird Name: {predicted_bird}')
        
        # Find bird image and info
        bird_images_folder = "static/images_of_birds/"
        bird_info = {
            'Acridotheres tristis': {
                "common_name": "Common Myna",
                "scientific_name": "Acridotheres tristis",
                "description": "A noisy bird with a distinctive yellow eye-patch and loud, mimicking calls. Often found in urban areas.",
                "locations": "Widespread across South Asia and introduced in other tropical and subtropical regions.",
                "life_span": "Acridotherestristis (Common Myna): 4-6 years",
                "migration": "Resident (non-migratory)",
                "season_wise": "Year-round in native range",
                "image_path" : r"static\images of birds\Acridotheres tristis.jfif"
            },
            'Aegithalos caudatus': {
                "common_name": "Long-tailed Tit",
                "scientific_name": "Aegithalos caudatus",
                "description": "A small passerine bird with a long tail and soft, fluffy plumage. Often seen in flocks.",
                "locations": "Found in woodlands across Europe and Asia.",
                "life_span": "Aegithaloscaudatus (Long-tailed Tit): 3-5 years",
                "migration": "Resident (non-migratory)",
                "season_wise": "Year-round in native range",
                "image_path" : r"static\images of birds\Aegithalos caudatus.jfif"
            },
            'Alauda arvensis': {
                "common_name": "Skylark",
                "scientific_name": "Alauda arvensis",
                "description": "Famous for its melodious song delivered in flight, often high in the sky.",
                "locations": "Found in open farmland and grasslands across Europe and Asia.",
                "life_span": "Alauda arvensis (Sky Lark): 3-5 years",
                "migration": "Partial migrant",
                "season_wise": "Migrates south during autumn (from northern Europe to southern Europe) and returns in spring",
                "image_path" : r"static\images of birds\Alauda arvensis.webp"
            },
            'Apus apus': {
                "common_name": "Common Swift",
                "scientific_name": "Apus apus",
                "description": "A migratory bird that spends most of its life in flight, with sickle-shaped wings.",
                "locations": "Breeds in Europe and Asia; winters in sub-Saharan Africa.",
                "life_span": "Apusapus (Common Swift): 4-5 years",
                "migration": "Migratory",
                "season_wise": "Migrates in the autumn from Europe to Africa and returns in spring",
                "image_path" : r"static\images of birds\Apus apus.webp"
            },
            'Cacicus cela': {
                "common_name": "Yellow-shouldered Blackbird",
                "scientific_name": "Cacicus cela",
                "description": "Identified by its striking yellow shoulders and melodious song.",
                "locations": "Native to tropical South America, particularly in the Amazon Basin.",
                "life_span": "Cacicuscela (Yellow-hooded Blackbird): 7-10 years",
                "migration": "Resident",
                "season_wise": "Year-round in South America",
                "image_path" : r"static\images of birds\Cacicus cela.jfif"
            },
            'Carduelis carduelis': {
                "common_name": "European Goldfinch",
                "scientific_name": "Carduelis carduelis",
                "description": "A small bird with a bright red face, black-and-white head, and yellow wing patch.",
                "locations": "Common across Europe, North Africa, and western Asia.",
                "life_span": "Cardueliscarduelis (European Goldfinch): 3-5 years",
                "migration": "Partial migrant",
                "season_wise": "Migrates in winter to southern parts of Europe and North Africa",
                "image_path" : r"static\images of birds\Carduelis carduelis.webp"
            },
            'Chloris chloris': {
                "common_name": "European Greenfinch",
                "scientific_name": "Chloris chloris",
                "description": "A chunky finch with a strong beak, predominantly green with yellow wing edges.",
                "locations": "Widespread across Europe, North Africa, and southwestern Asia.",
                "life_span": "Chlorischloris (Greenfinch): 2-3 years",
                "migration": "Partial migrant",
                "season_wise": "Moves southward in winter (from Northern Europe to Southern Europe)",
                "image_path" : r"static\images of birds\Chloris chloris.webp"
            },
            'Coccothraustes coccothraustes': {
                "common_name": "Hawfinch",
                "scientific_name": "Coccothraustes coccothraustes",
                "description": "A robust bird with a powerful beak, known for cracking hard seeds.",
                "locations": "Found in deciduous forests of Europe and Asia.",
                "life_span": "Coccothraustescoccothraustes (Hawfinch): 5-10 years",
                "migration": "Partial migrant",
                "season_wise": "Migrates in winter from northern to southern Europe",
                "image_path" : r"static\images of birds\Coccothraustes coccothraustes.webp"
            },
            'Columba livia': {
                "common_name": "Rock Pigeon",
                "scientific_name": "Columba livia",
                "description": "The wild ancestor of domestic pigeons, often found in urban areas and cliffs.",
                "locations": "Widespread globally, especially in cities and towns.",
                "life_span": "Coccothraustescoccothraustes (Hawfinch): 5-10 years",
                "migration": "Partial migrant",
                "season_wise": "Migrates in winter from northern to southern Europe",
                "image_path" : r"static\images of birds\Columba livia.webp"
            },
                'Corvus corone': {
                "common_name": "Carrion Crow",
                "scientific_name": "Corvus corone",
                "description": "A large black bird known for its intelligence and adaptability to various environments.",
                "locations": "Widespread across Europe and parts of Asia.",
                "life_span": "Corvus corone (Carrion Crow): 7-14 years",
                "migration": "Resident (non-migratory)",
                "season_wise": "Year-round in Europe, Asia, and parts of Africa",
                "image_path" : r"static\images of birds\Corvus corone.webp"
            },
            'Corvus corone': {
                "common_name": "Carrion Crow",
                "scientific_name": "Corvus corone",
                "description": "A large black bird known for its intelligence and adaptability to various environments.",
                "locations": "Widespread across Europe and parts of Asia.",
                "life_span": "Corvuscorone (Carrion Crow): 7-14 years",
                "migration": "Resident (non-migratory)",
                "season_wise": "Year-round in Europe, Asia, and parts of Africa",
                "image_path" : r"static\images of birds\Corvus corone.webp"
            },
            #######################
            'Corvus frugilegus': {
                    "common_name": "Rook",
                    "scientific_name": "Corvus frugilegus",
                    "description": "A social bird, often seen in large flocks, with glossy black feathers and pale, bare skin around the beak.",
                    "locations": "Common in farmlands and open woodlands across Europe and Asia.",
                    "life_span": "Corvusfrugilegus (Rook): 5-15 years",
                    "migration": "Partial migrant",
                    "season_wise": "Migrates from northern Europe to more temperate regions in winter",
                    "image_path": r"static\images of birds\Corvus frugilegus.webp"
                },
                'Cuculus canorus': {
                    "common_name": "Common Cuckoo",
                    "scientific_name": "Cuculus canorus",
                    "description": "Known for its parasitic breeding strategy, laying eggs in other birds’ nests.",
                    "locations": "Inhabits forests and open countryside across Europe and Asia.",
                    "life_span": "Cuculuscanorus (Common Cuckoo): 4-7 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to sub-Saharan Africa in autumn, returning in spring",
                    "image_path": r"static\images of birds\Cuculus canorus.webp"
                },
                'Delichon urbicum': {
                    "common_name": "House Martin",
                    "scientific_name": "Delichon urbicum",
                    "description": "A small migratory bird that builds mud nests on buildings, with white underparts and dark glossy blue above.",
                    "locations": "Breeds in Europe and winters in Africa.",
                    "life_span": "Delichonurbicum (House Martin): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa in winter, returning in spring",
                    "image_path": r"static\images of birds\Delichon urbicum.webp"
                },
                'Dendrocopos major': {
                    "common_name": "Great Spotted Woodpecker",
                    "scientific_name": "Dendrocopos major",
                    "description": "A medium-sized woodpecker with striking black, white, and red plumage, known for its drumming.",
                    "locations": "Found in woodlands across Europe and northern Asia.",
                    "life_span": "Dendrocoposmajor (Great Spotted Woodpecker): 4-11 years",
                    "migration": "Partial migrant",
                    "season_wise": "Some populations move to lower altitudes or southern regions in winter",
                    "image_path": r"static\images of birds\Dendrocopos major.webp"
                },
                'Dumetella carolinensis': {
                    "common_name": "Eastern Towhee",
                    "scientific_name": "Dumetella carolinensis",
                    "description": "A large sparrow with bold black, white, and orange patterning, known for its 'drink-your-tea' song.",
                    "locations": "Common in forests and gardens in eastern North America.",
                    "life_span": "Dumetellacarolinensis (Gray Catbird): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from North America to Central America and the Caribbean in autumn",
                    "image_path": r"static\images of birds\Dumetella carolinensis.jfif"
                },
                'Emberiza citrinella': {
                    "common_name": "Yellowhammer",
                    "scientific_name": "Emberiza citrinella",
                    "description": "A bright yellow-headed bird with a distinctive song that resembles 'a little bit of bread and no cheese.'",
                    "locations": "Found in farmlands, grasslands, and woodland edges across Europe and Asia.",
                    "life_span": "Emberizacitrinella (Yellowhammer): 3-5 years",
                    "migration": "Partial migrant",
                    "season_wise": "Migrates in autumn from northern Europe to southern Europe",
                    "image_path": r"static\images of birds\Emberiza citrinella.webp"
                },
                'Erithacus rubecula': {
                    "common_name": "European Robin",
                    "scientific_name": "Erithacus rubecula",
                    "description": "A small bird with a distinctive orange-red breast, known for its friendly nature and territorial singing.",
                    "locations": "Found in gardens, woodlands, and parks across Europe.",
                    "life_span": "Erithacusrubecula (European Robin): 2-3 years",
                    "migration": "Partial migrant",
                    "season_wise": "Migrates from northern Europe to southern Europe and North Africa in winter",
                    "image_path": r"static\images of birds\Erithacus rubecula.webp"
                },
                'Ficedula hypoleuca': {
                    "common_name": "European Pied Flycatcher",
                    "scientific_name": "Ficedula hypoleuca",
                    "description": "A small migratory bird with striking black and white plumage, known for catching insects in flight.",
                    "locations": "Breeds in Europe and migrates to Africa for winter.",
                    "life_span": "Ficedulahypoleuca (Willow Tit): 2-3 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa for winter",
                    "image_path": r"static\images of birds\Ficedula hypoleuca.webp"
                },
                'Fringilla coelebs': {
                    "common_name": "Common Chaffinch",
                    "scientific_name": "Fringilla coelebs",
                    "description": "A colorful finch, males have a pinkish face and blue-grey crown, while females are more subdued in color.",
                    "locations": "Found in forests, gardens, and parks across Europe, North Africa, and western Asia.",
                    "life_span": "Fringillacoelebs (Chaffinch): 3-5 years",
                    "migration": "Partial migrant",
                    "season_wise": "Migrates south in winter from northern Europe to southern Europe",
                    "image_path": r"static\images of birds\Fringilla coelebs.webp"
                },
                'Gallus gallus': {
                    "common_name": "Red Junglefowl",
                    "scientific_name": "Gallus gallus",
                    "description": "The wild ancestor of domestic chickens, with vivid plumage and loud crowing calls.",
                    "locations": "Found in forests and scrublands across Southeast Asia.",
                    "life_span": "Gallusgallus (Red Junglefowl): 5-10 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in Southeast Asia",
                    "image_path": r"static\images of birds\Gallus gallus.webp"
                },
                'Garrulus glandarius': {
                    "common_name": "Eurasian Jay",
                    "scientific_name": "Garrulus glandarius",
                    "description": "A colorful bird with striking blue wing feathers, known for its role in dispersing acorns.",
                    "locations": "Inhabits forests and woodlands across Europe and Asia.",
                    "life_span": "Garrulusglandarius (Eurasian Jay): 3-5 years",
                    "migration": "Partial migrant",
                    "season_wise": "Some populations move to lower altitudes in winter",
                    "image_path": r"static\images of birds\Garrulus glandarius.webp"
                },
                'Hirundo rustica': {
                    "common_name": "Barn Swallow",
                    "scientific_name": "Hirundo rustica",
                    "description": "A migratory bird with a deeply forked tail, known for its swift, agile flight.",
                    "locations": "Breeds in Europe and Asia; winters in Africa.",
                    "life_span": "Hirundorustica (Barn Swallow): 4-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe and North America to Africa and South America",
                    "image_path": r"static\images of birds\Hirundo rustica.webp"
                },
                'Lanius excubitor': {
                    "common_name": "Northern Shrike",
                    "scientific_name": "Lanius excubitor",
                    "description": "A predatory songbird known for impaling its prey on thorns, also called the 'butcher bird.'",
                    "locations": "Found in open habitats across Europe, Asia, and North America.",
                    "life_span": "Laniusexcubitor (Great Grey Shrike): 4-5 years",
                    "migration": "Partial migrant",
                    "season_wise": "Moves from northern regions to more temperate areas in winter",
                    "image_path": r"static\images of birds\Lanius excubitor.webp"
                },
                'Luscinia luscinia': {
                    "common_name": "Nightingale",
                    "scientific_name": "Luscinia luscinia",
                    "description": "Famed for its beautiful and complex song, often heard during the night.",
                    "locations": "Breeds in Europe and winters in sub-Saharan Africa.",
                    "life_span": "Luscinialuscinia (Nightingale): 2-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa in winter",
                    "image_path": r"static\images of birds\Luscinia luscinia.jfif"
                },
                'Motacilla alba': {
                    "common_name": "White Wagtail",
                    "scientific_name": "Motacilla alba",
                    "description": "A slender bird with a characteristic wagging tail, often found near water.",
                    "locations": "Common across Europe, Asia, and parts of North Africa.",
                    "life_span": "Motacillaalba (White Wagtail): 2-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from northern Europe to southern Europe and parts of Asia",
                    "image_path": r"static\images of birds\Motacilla alba.jfif"
                },
                'Motacilla flava': {
                    "common_name": "Yellow Wagtail",
                    "scientific_name": "Motacilla flava",
                    "description": "A small bird with bright yellow underparts and a constant wagging tail.",
                    "locations": "Breeds in Europe and Asia; winters in Africa.",
                    "life_span": "Motacillaflava (Yellow Wagtail): 2-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa and Southern Asia",
                    "image_path": r"static\images of birds\Motacilla flava.jfif"
                },
                'Parus major': {
                    "common_name": "Great Tit",
                    "scientific_name": "Parus major",
                    "description": "A large tit species with a bold black head and white cheeks, known for its varied calls.",
                    "locations": "Found in woodlands, gardens, and parks across Europe and Asia.",
                    "life_span": "Parusmajor (Great Tit): 2-3 years",
                    "migration": "Partial migrant",
                    "season_wise": "Some populations migrate short distances in winter",
                    "image_path": r"static\images of birds\Parus major.jfif"
                },
                'Passer domesticus': {
                    "common_name": "House Sparrow",
                    "scientific_name": "Passer domesticus",
                    "description": "A small, social bird that often lives close to human settlements.",
                    "locations": "Widespread in urban and rural areas across the world.",
                    "life_span": "Passerdomesticus (House Sparrow): 3 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in most areas",
                    "image_path": r"static\images of birds\Passer domesticus.jfif"
                },
                'Phoenicurus ochruros': {
                    "common_name": "Black Redstart",
                    "scientific_name": "Phoenicurus ochruros",
                    "description": "A small bird with black and red plumage, often seen perched on buildings or rocky outcrops.",
                    "locations": "Common in urban areas and mountains across Europe and Asia.",
                    "life_span": "Phoenicurusochruros (Black Redstart): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from northern Europe to southern Europe and North Africa",
                    "image_path": r"static\images of birds\Phoenicurus ochruros.jfif"
                },
                'Phoenicurus phoenicurus': {
                    "common_name": "Common Redstart",
                    "scientific_name": "Phoenicurus phoenicurus",
                    "description": "A striking bird with orange-red tail and chest, known for its flicking tail movements.",
                    "locations": "Breeds in Europe and western Asia; winters in Africa.",
                    "life_span": "Phoenicurusphoenicurus (Common Redstart): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa in winter",
                    "image_path": r"static\images of birds\Phoenicurus phoenicurus.jfif"
                },
                'Phylloscopus collybita': {
                    "common_name": "Common Chiffchaff",
                    "scientific_name": "Phylloscopus collybita",
                    "description": "A small, greenish-brown warbler known for its repetitive 'chiff-chaff' song.",
                    "locations": "Found in woodlands and gardens across Europe and Asia.",
                    "life_span": "Phylloscopuscollybita (Chiffchaff): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to southern Europe and North Africa",
                    "image_path": r"static\images of birds\Phylloscopus collybita.jfif"
                },
                'Phylloscopus trochilus': {
                    "common_name": "Willow Warbler",
                    "scientific_name": "Phylloscopus trochilus",
                    "description": "A small migratory bird with a gentle, melodic song, often confused with the Chiffchaff.",
                    "locations": "Breeds in Europe and migrates to Africa.",
                    "life_span": "Phylloscopustrochilus (Willow Warbler): 3-5 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa",
                    "image_path": r"static\images of birds\Phylloscopus trochilus.jfif"
                },
                'Pica pica': {
                    "common_name": "Eurasian Magpie",
                    "scientific_name": "Pica pica",
                    "description": "A large, intelligent bird with striking black-and-white plumage and a long tail.",
                    "locations": "Widespread across Europe, Asia, and North Africa.",
                    "life_span": "Picapica (Magpie): 4-8 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round",
                    "image_path": r"static\images of birds\Pica pica.jfif"
                },
                'Pycnonotus cafer': {
                    "common_name": "Red-vented Bulbul",
                    "scientific_name": "Pycnonotus cafer",
                    "description": "A medium-sized songbird with a distinctive black crest and red vent.",
                    "locations": "Native to South Asia but introduced in many other regions.",
                    "life_span": "Pycnonotuscafer (Black-crowned Night Heron): 10-15 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in native areas",
                    "image_path": r"static\images of birds\Pycnonotus cafer.jfif"
                },
                'Pycnonotus jocosus': {
                    "common_name": "Red-whiskered Bulbul",
                    "scientific_name": "Pycnonotus jocosus",
                    "description": "A bird with a striking crest and red patches on its cheeks, known for its cheerful call.",
                    "locations": "Found in forests and gardens across South and Southeast Asia.",
                    "life_span": "Pycnonotusjocosus (Red-whiskered Bulbul): 6-10 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in native areas",
                    "image_path": r"static\images of birds\Pycnonotus jocosus.jfif"
                },
                'Sitta europaea': {
                    "common_name": "Eurasian Nuthatch",
                    "scientific_name": "Sitta europaea",
                    "description": "A small bird known for its ability to climb down trees headfirst, with a blue-grey back and chestnut underparts.",
                    "locations": "Inhabits woodlands across Europe and parts of Asia.",
                    "life_span": "Sittaeuropaea (Eurasian Nuthatch): 3-5 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round",
                    "image_path": r"static\images of birds\Sitta europaea.jfif"
                },
                'Streptopelia turtur': {
                    "common_name": "European Turtle Dove",
                    "scientific_name": "Streptopelia turtur",
                    "description": "A migratory dove with a gentle purring call, characterized by its mottled brown feathers.",
                    "locations": "Breeds in Europe and winters in Africa.",
                    "life_span": "Streptopeliaturtur (Turtle Dove): 2-4 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa in autumn",
                    "image_path": r"static\images of birds\Streptopelia turtur.jfif"
                },
                'Sturnus vulgaris': {
                    "common_name": "Common Starling",
                    "scientific_name": "Sturnus vulgaris",
                    "description": "A highly social bird with iridescent plumage and a talent for mimicking sounds.",
                    "locations": "Widespread across Europe, Asia, and introduced to North America.",
                    "life_span": "Sturnusvulgaris (Common Starling): 2-3 years",
                    "migration": "Partial migrant",
                    "season_wise": "Some populations migrate south in winter",
                    "image_path": r"static\images of birds\Sturnus vulgaris.jfif"
                },
                'Troglodytes troglodytes': {
                    "common_name": "Eurasian Wren",
                    "scientific_name": "Troglodytes troglodytes",
                    "description": "A tiny bird with a loud voice and a habit of flicking its short tail.",
                    "locations": "Found in dense undergrowth across Europe, Asia, and North Africa.",
                    "life_span": "Troglodytestroglodytes (Winter Wren): 2-3 years",
                    "migration": "Partial migrant",
                    "season_wise": "Moves from northern Europe to more temperate regions during winter",
                    "image_path": r"static\images of birds\Troglodytes troglodytes.jfif"
                },
                'Turdus merula': {
                    "common_name": "Common Blackbird",
                    "scientific_name": "Turdus merula",
                    "description": "A familiar bird with glossy black plumage (in males) and a melodious song.",
                    "locations": "Widespread across Europe, Asia, and North Africa.",
                    "life_span": "Turdusmerula (European Blackbird): 2-3 years",
                    "migration": "Partial migrant",
                    "season_wise": "Some populations migrate south in winter",
                    "image_path": r"static\images of birds\Turdus merula.jfif"
                },
                'Turdus philomelos': {
                    "common_name": "Song Thrush",
                    "scientific_name": "Turdus philomelos",
                    "description": "A medium-sized thrush with a beautiful, varied song and a speckled chest.",
                    "locations": "Found in woodlands, gardens, and parks across Europe and western Asia.",
                    "life_span": "Turdusphilomelos (Song Thrush): 2-4 years",
                    "migration": "Partial migrant",
                    "season_wise": "Migrates south during winter, particularly from northern Europe",
                    "image_path": r"static\images of birds\Turdus philomelos.jfif"
                },
                'Turdus pilaris': {
                    "common_name": "Fieldfare",
                    "scientific_name": "Turdus pilaris",
                    "description": "A large thrush with grey head and rump, known for forming flocks in the winter.",
                    "locations": "Breeds in northern Europe and Asia; winters in milder regions.",
                    "life_span": "Turduspilaris (American Robin): 2-3 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from northern Europe to southern Europe in winter",
                    "image_path": r"static\images of birds\Turdus pilaris.jfif"
                },
                'Upupa epops': {
                    "common_name": "Eurasian Hoopoe",
                    "scientific_name": "Upupa epops",
                    "description": "A striking bird with a fan-shaped crest and a long, curved bill, known for its distinctive 'oop-oop-oop' call.",
                    "locations": "Found in open landscapes across Europe, Asia, and North Africa.",
                    "life_span": "Upupaepops (Eurasian Hoopoe): 5-9 years",
                    "migration": "Migratory",
                    "season_wise": "Migrates from Europe to Africa in winter",
                    "image_path": r"static\images of birds\Upupa epops.jfif"
                },
                'Variegated Tinamou': {
                    "common_name": "Variegated Tinamou",
                    "scientific_name": "Crypturellus variegatus",
                    "description": "A ground-dwelling bird known for its distinctively patterned plumage, often found in forested areas.",
                    "locations": "Found in tropical lowland forests across Central and South America.",
                    "life_span": "Variegated Tinamou (sound): 12-20 years (In the wild)",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in tropical lowlands",
                    "image_path": r"static\images of birds\Variegated Tinamou_sound.jfif"
                },
                'Andean Tinamou': {
                    "common_name": "Andean Tinamou",
                    "scientific_name": "Nothoprocta pentlandii",
                    "description": "A robust bird adapted to high-altitude environments, known for its cryptic coloration.",
                    "locations": "Inhabits grasslands and shrublands in the Andes mountains.",
                    "life_span": "Andean Tinamou (sound): 10-15 years (In the wild)",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in the Andes",
                    "image_path": r"static\images of birds\Andean Tinamou_sound.jfif"
                },
                'Andean Guan': {
                    "common_name": "Andean Guan",
                    "scientific_name": "Penelope montagnii",
                    "description": "A large, dark bird with a distinctive crest, known for its loud calls and preference for forest habitats.",
                    "locations": "Primarily found in the Andean forests of Colombia and Ecuador.",
                    "life_span": "Andean Guan (sound): 15-20 years (In the wild)",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in the Andes",
                    "image_path": r"static\images of birds\Andean Guan_sound.jfif"
                },
                'Band-tailed Guan': {
                    "common_name": "Band-tailed Guan",
                    "scientific_name": "Penelope superciliaris",
                    "description": "A medium to large bird with a distinctive band on its tail, often found in tropical and subtropical forests.",
                    "locations": "Found in the Andean foothills and lowland forests of Colombia and Ecuador.",
                    "life_span": "Band-tailed Guan (sound): 10-15 years (In the wild)",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in the Andean foothills",
                    "image_path": r"static\images of birds\Band-tailed Guan_sound.jfif"
                },
                'Cauca Guan': {
                    "common_name": "Cauca Guan",
                    "scientific_name": "Penelope perspicax",
                    "description": "A large, forest-dwelling bird with striking plumage and a distinctive call.",
                    "locations": "Endemic to the Cauca Valley in Colombia, inhabiting tropical lowland forests.",
                    "life_span": "Cauca Guan (sound): 15-20 years (In the wild)",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in the Cauca Valley",
                    "image_path": r"static\images of birds\Cauca Guan_sound.jfif"
                },
                'East Brazilian Chachalaca': {
                    "common_name": "East Brazilian Chachalaca",
                    "scientific_name": "Ortalis guttata",
                    "description": "A medium-sized bird known for its loud, chachalaca call and preference for dense forests.",
                    "locations": "Native to eastern Brazil, particularly in coastal forests.",
                    "life_span": "East Brazilian Chachalaca (sound): 10-15 years",
                    "migration": "Non-migratory",
                    "season_wise": "Year-round in coastal forests of Brazil",
                    "image_path": r"static\images of birds\East Brazilian Chachalaca_sound.jfif"
                }


        }
        
        # If the bird is in the bird_info dictionary, retrieve its data
        if predicted_bird in bird_info:
            bird_data = bird_info[predicted_bird]
            common_name = bird_data["common_name"]
            print(1111111111,common_name)
            scientific_name = bird_data["scientific_name"]
            print(2222222222,scientific_name)
            description = bird_data["description"]
            print(3333333333,description)
            locations = bird_data["locations"]
            print(4444444444,locations)
            life_span=bird_data["life_span"]
            migration=bird_data["migration"]
            season_wise=bird_data["season_wise"]
            image_path = bird_data["image_path"]
            print(5555555555,image_path)
            # image_path = os.path.join(bird_images_folder, f"{predicted_bird}.jpg")
        else:
            common_name = "Unknown Bird"
            scientific_name = ""
            description = "No description available."
            locations = "Unknown"
            life_span=""
            migration=""
            season_wise=""
            image_path = ""

        # Render the template with the bird image, common name, scientific name, description, and locations
        return render_template(
            'upload.html',
            prediction=predicted_bird,
            common_name=common_name,
            scientific_name=scientific_name,
            description=description,
            locations=locations,
            life_span=life_span,
            migration=migration,
            season_wise=season_wise,
            image_path=image_path,
            path=mypath
        )
    
    # Handle GET request - render the upload page with a default message
    return render_template("upload.html", message="Please upload an audio file")



if __name__ == '__main__':
    app.run(debug = True)