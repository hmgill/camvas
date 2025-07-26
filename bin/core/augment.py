import tensorflow as tf

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=0):
        super().__init__()
        self.seed = seed

        # Initialize image augmentation layers
        self.image_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomBrightness(
                0.1,
                value_range=(0, 1.0)
            ),
            tf.keras.layers.GaussianNoise(0.1),
            tf.keras.layers.RandomRotation(
                (-0.2, 0.3),
                fill_mode='constant',
                interpolation='nearest',
                fill_value=1,
                seed=self.seed
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.2,
                width_factor=0.2,
                fill_value=1,
                fill_mode='reflect',
                interpolation='nearest',
                seed=self.seed,
            ),
            tf.keras.layers.RandomZoom(
                (-0.75, 0.2),
                fill_mode="reflect",
                fill_value=1,
                interpolation="nearest",
                seed=self.seed
            ),
        ])

        # Initialize mask augmentation layers
        self.mask_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(
                (-0.2, 0.3),
                fill_mode='constant',
                interpolation='nearest',
                fill_value=0,
                seed=self.seed
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.2,
                width_factor=0.2,
                fill_value=0,
                fill_mode='reflect',
                interpolation='nearest',
                seed=self.seed,
            ),
            tf.keras.layers.RandomZoom(
                (-0.75, 0.2),
                fill_mode="reflect",
                fill_value=0,
                interpolation="nearest",
                seed=self.seed
            ),
        ])

    def call(self, images, masks):
        images = self.image_augmentation(images)
        masks = self.mask_augmentation(masks)
        masks = tf.where(masks > 0.118, 1, 0)
        return images, masks
