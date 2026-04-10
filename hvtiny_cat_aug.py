# HVTiny CNN
# Low complexity yet accurate model - well suited for MCU-oriented TinyML apps.
# Copyright Radu DOGARU  radu_d@ieee.org , august-october 2025

# Last updated Mar 26, 2026 - for sparse coding 
#----------------------------------------------------------------------

def hvtiny(input_shape, num_classes, profil=[1, 3, 2], k=60, nl=(1,1), hid=True, flat=0, scale=True, drop=0.3, lr=0.001) :
        import keras 
        kernel_size = (3,3)
        pool_size = (4,4)
        pool_strides = (2,2)
        c=len(profil)-1
        # updated nov. 16 - including nl (original model used nl=2)
        # and automatically adjusted c from the size of profil[]
        print('Profil: ',str(profil))
        drop1=drop
        pad='same'

        if scale:
            inputs = keras.Input(shape=input_shape) / 255 #
        else:
            inputs = keras.Input(shape=input_shape)
    	n = int(k)
	#
        x=data_augmentation(inputs)
        # convolutional macro-block 0
        x = keras.layers.Conv2D(n, kernel_size, padding=pad)(x)  # Separable
        for i in range(1, nl[0]+1) :
          print('adding')
          x = keras.layers.ReLU()(x)
          x = keras.layers.DepthwiseConv2D(kernel_size, padding=pad)(x)

        #x=keras.layers.BatchNormalization()(x)                        # May help in some cases
        x=keras.layers.MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding=pad)(x)
        if drop1>0:
            x=keras.layers.Dropout(drop1)(x)
        number_of_cells_limited = False

        #adding c macro-blocks 1,2 ... c
        for i in range(1, c + 1) :
            if x.shape[1] <= 1 or x.shape[2] <= 1 :
                number_of_cells_limited = True
                break;

            n=int(k*profil[i])
            x = keras.layers.SeparableConv2D(n, kernel_size, padding=pad)(x)
            if i==1:
              for j in range(1, nl[1]+1) :
                x = keras.layers.ReLU()(x)
                x = keras.layers.DepthwiseConv2D(kernel_size, padding=pad)(x)
            x=keras.layers.MaxPooling2D(pool_size=pool_size,strides=pool_strides,padding=pad)(x)
            if drop1>0:
                x=keras.layers.Dropout(drop1)(x)

        # Output classifier
        if flat==0:
            x = keras.layers.GlobalAveragePooling2D()(x)
        else:
            x= keras.layers.Flatten()(x)
        if hid:   # original model in the paper has hid=False
          x = keras.layers.Dense(n, activation='relu')(x)  # may increase accuracy + extra resources
        x = keras.layers.BatchNormalization()(x)
        x=  keras.layers.Dropout(rate=0.4)(x)
        x = keras.layers.Dense(num_classes)(x)
        outputs = keras.layers.Softmax()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        opt = keras.optimizers.Adam(learning_rate=lr) 

        model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return model
