from matplotlib import pyplot
import numpy as np
import math

gsum = 0.
gvar = 0.
gcount = 0

def show_images(gen, compare=False, find_mismatch=False):
    first = True
    pyplot.ion()
    (fig, axs) = pyplot.subplots(3, 3)
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')
    pyplot.show()
    norm=None
    # norm=matplotlib.colors.Normalize()
    # norm=matplotlib.colors.NoNorm()
    while True:
        global gsum, gvar, gcount
        got_mismatch = False
        for i in range(0, 9):
        # for i in range(0, 8):
            if compare:
                (x, y, yp) = next(gen)
            else:
                (x, y) = next(gen)
                if first:
                    yp = y
            scount = 224*224
            ssum = x.sum()
            smean = ssum / scount
            svar = np.var(x)
            if gcount:
                gmean = gsum / gcount
                delta = smean - gmean
                m_a = gvar*(gcount-1)
                m_b = svar*(scount-1)
                M2 = m_a + m_b + delta**2 * gcount * scount / (gcount + scount)
            else:
                M2 = ssum
            gsum += ssum
            gcount += scount
            gvar = M2 / (gcount - 1)
            c = (y < 0.5)
            cp = (yp < 0.5)
            first = False
            axs[i].imshow(np.squeeze(x), cmap='gray', norm=norm)
            if c:
                title = "Match"
                color="green"
            else:
                title = "No match"
                color="black"
            # title=str(np.mean(x))[:4]
            if compare:
                score = int(100*(1.0-yp))
            else:
                score = int(100*(1.0-y))
            # mean = np.mean(x)
            # score=int(100.0*mean)
            if (c != cp):
                got_mismatch = True
                if compare:
                    color = "red"
                    if cp:
                        title = "False POS"
                    else:
                        title = "False NEG"
            title += " [" +str(score) + "]"
            axs[i].set_title(title, color=color)
        # x = preprocess(x)
        # axs[8].imshow(np.squeeze(x), cmap='gray', norm=norm)
        print("#={0}, mean={1}, std={2}".format(gcount, gsum/gcount, math.sqrt(gvar)))
        if got_mismatch or not find_mismatch:
            pyplot.waitforbuttonpress()
