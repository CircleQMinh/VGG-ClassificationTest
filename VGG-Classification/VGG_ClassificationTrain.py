import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, InputLayer
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder,filename), target_size=(224, 224))
        if img is not None:
            images.append(img)
    return images

def PredictMyImg(image):

    img=image

    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)

    yhat = model.predict(image)

    label = decode_predictions(yhat,top=10)

    name = []
    prob = []
    print("The result is :")
    for lb in label:
        for pred in lb:
            print('%s (%.2f%%)' % (pred[1], pred[2]*100))
            name.append(pred[1])
            prob.append(pred[2]*100)

    fig, ax = plt.subplots(figsize =(12, 6))

    ax.barh(name, prob)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_ticks_position('default')
    ax.yaxis.set_ticks_position('default')

    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10) 

    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2) 

    ax.invert_yaxis() 

    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey') 

    ax.set_title('Result',
                 loc ='left', )

    fig.patch.set_facecolor('lightblue')
    ax.set_facecolor('white')
    newax = fig.add_axes([0.65, 0.5, 0.3, 0.3], anchor='NE', zorder=-1)
    newax.imshow(img)
    newax.set_title('Picture',
                 loc ='left', )
    newax.axis('off')

    plt.subplots_adjust(right=0.65)
    plt.grid()
    plt.show()















#start here !!!
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
modelVGG16 = VGG16()
modelVGG19 = VGG19()
print(modelVGG16.summary())
print(modelVGG19.summary())












# 1 s??? th??ng s??? c???a h??m VGG16()

#include_top (True): C?? bao g???m c??c l???p ?????u ra(3 fully-connected layers) cho m?? h??nh hay kh??ng. 
#weights (???imagenet???): Tr???ng s??? c???n t???i. B???n c?? th??? ch??? ?????nh 1 weight n???u ????? tr???ng s??? m???c ?????nh l?? 'imagenet'
#input_tensor (Kh??ng c??): L???p ?????u v??o m???i n???u b???n c?? ?? ?????nh ??i???u ch???nh m?? h??nh
# tr??n d??? li???u m???i c?? k??ch th?????c kh??c.
#input_shape (Kh??ng c??): K??ch th?????c h??nh ???nh m?? m?? h??nh d??? ki???n s??? nh???n n???u 
#b???n thay ?????i l???p ?????u v??o.
#pooling  (Kh??ng c??): Lo???i pooling  ????? s??? d???ng khi b???n ??ang ????o t???o m???t t???p 
#h???p c??c l???p ?????u ra m???i. (avg - max)
#classes (1000): S??? l???p (v?? d???: k??ch th?????c c???a vect?? ?????u ra) cho m?? h??nh.









#demo d??? ??o??n ???nh v???i vgg



model=modelVGG16
folder="Good"
MyImg=load_images_from_folder(folder)
for img in MyImg:
    PredictMyImg(img)
























def createModelVGG16():
    model=Sequential()
    model.add(InputLayer(input_shape=(224,224,3),name='input_1'))
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    classes=1000
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


def createModelVGG19():
    model=Sequential()
    model.add(InputLayer(input_shape=(224,224,3),name='input_1'))
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    classes=1000
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model





#model=createModelVGG16()
#from keras.preprocessing.image import ImageDataGenerator 

#img_width=224
#img_height=224

#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   horizontal_flip = True)
#training_set = train_datagen.flow_from_directory('Train',
#                                                 target_size = (img_width, img_height),
#                                                 batch_size = 32,
#                                                 class_mode = 'categorical')
#
#test_datagen = ImageDataGenerator(rescale = 1./255)
#test_set = test_datagen.flow_from_directory('Test',
#                                            target_size = (img_width, img_height),
#                                            batch_size = 32,
#                                            class_mode = 'categorical')

#history=model.fit(x = training_set, validation_data = test_set, epochs =  20)



















