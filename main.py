import cv2
import face_recognition as fr

imgbase = fr.load_image_file('deanmorgan.png')
imgbase = cv2.cvtColor(imgbase, cv2.COLOR_BGR2RGB)
imgcomp = fr.load_image_file('deanmorgan2.png')
imgcomp = cv2.cvtColor(imgcomp, cv2.COLOR_BGRA2BGR)


faceLoc = fr.face_locations(imgbase)[0]
cv2.rectangle(imgbase, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
print(faceLoc)

encodeBase = fr.face_encodings(imgbase)[0]
encodeComp = fr.face_encodings(imgcomp)[0]

comparacao = fr.compare_faces([encodeBase], encodeComp)

print(comparacao)
cv2.imshow('Imagem Base', imgbase)
cv2.imshow('Imagem de Comparacao', imgcomp)
cv2.waitKey(0)
