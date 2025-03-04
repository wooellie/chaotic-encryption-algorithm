import logisticKey as key   # Importing the key generating function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import hashlib

# Accepting an image
path = str(input('Enter the path of image\n'))
image = img.imread(path)

# Displaying the image
plt.imshow(image)
plt.show()

# Generating dimensions of the image
height = image.shape[0]
width = image.shape[1]
print(height, width)#height, width表示图像的大小

# Generating keys
# Calling logistic_key and providing r value such that the keys are pseudo-random
# and generating a key for every pixel of the image
#######################检查加密、解密过程中的密钥一致性##############################
x_logistice = float(input("please enter logistic initial value for encryption: "))
y_tente = float(input("please enter tent initial value for encryption: "))
r_logistice = float(input("please enter logistic parameter for encryption: "))
r_tente = float(input("please enter tent parameter for encryption: "))
generatedKeye = key.logistic_tent_key(x_logistice, y_tente, r_logistice, r_tente, height*width) #height*width表示密钥长度等于图像的总像素数，每个像素对应一个密钥值。
print(generatedKeye)
#======================计算密钥的sha-256哈希值=======================================
key_bytese = bytes(generatedKeye)#将密钥列表转换为字节
key_hashe = hashlib.sha256(key_bytese).hexdigest()
print(f"[important] please notedown the hash-key for decryption.{key_hashe}")

# Encryption using XOR
z = 0

# Initializing the encrypted image
encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# Substituting all the pixels in original image with nested for
for i in range(height):
    for j in range(width):
        # USing the XOR operation between image pixels and keys
#######################################################################################################
        encryptedImage[i, j] = image[i, j].astype(int) ^ generatedKeye[z]
        z += 1

# Displaying the encrypted image
plt.imshow(encryptedImage)
plt.show()

#=========================检查加密、解密过程中的密钥一致性===================================================
x_logisticd = float(input("please enter logistic initial value for decryption: "))
y_tentd = float(input("please enter tent initial value for decryption: "))
r_logisticd = float(input("please enter logistic parameter for decryption: "))
r_tentd = float(input("please enter tent parameter for decryption: "))
generatedKeyd = key.logistic_tent_key(x_logisticd, y_tentd, r_logisticd, r_tentd, height*width) #height*width表示密钥长度等于图像的总像素数，每个像素对应一个密钥值。
print(generatedKeyd)
#======================重新计算解密密钥的sha-256哈希值=======================================
key_bytesd = bytes(generatedKeyd)#将密钥列表转换为字节
key_hashd = hashlib.sha256(key_bytesd).hexdigest()
original_hash = input("please enter the hash_key when encryption: ")
if original_hash != key_hashd:
    print("hash_key mismatch. possible reason: wrong parameters input")
else :
    print("hash_key matches. the image is going to decrypt.")


# Decryption using XOR
z = 0

# Initializing the decrypted image
decryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# Substituting all the pixels in encrypted image with nested for
for i in range(height):
    for j in range(width):
        # USing the XOR operation between encrypted image pixels and keys
######################################################################################################
        decryptedImage[i, j] = encryptedImage[i, j].astype(int) ^ generatedKeyd[z]
        z += 1

# Displaying the decrypted image
#plt.imshow(decryptedImage)
#plt.show()
