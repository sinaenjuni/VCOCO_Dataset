from random import randrange


img_size = (427, 640)

for i in range(10):
    xmin = randrange(0, img_size[0]-400)
    ymin = randrange(0, img_size[1]-400)
    xmax = xmin+400
    ymax = ymin+400

    # sample_list.append(img.crop((x1, y1, x1 + matrix, y1 + matrix)))
    print(xmax - xmin, ymax - ymin)