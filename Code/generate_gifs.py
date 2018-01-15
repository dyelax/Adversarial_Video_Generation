import constants as c
import os
import sys
import imageio
import getopt

IMGAE_SUFFIX = '.png'
GIF_SUFFIX = '.gif'
GIF_DURATION = 0.2


def images_to_gif(root_dir, input_filenames, output_filename):
    with imageio.get_writer(os.path.join(root_dir, output_filename), mode='I') as writer:
        writer.duration = GIF_DURATION
        for filename in input_filenames:
            image = imageio.imread(os.path.join(root_dir, filename))
            writer.append_data(image)
        print('GIF GENERATED:', output_filename, '-', len(input_filenames), 'frames')
        print('FILES:', input_filenames)


def check_numbers(names):
    assert(names)
    title = names[0][0]
    i = 0
    j = 0
    while i < len(names):
        if names[i][1] == j:
            i += 1
            j += 1
            continue
        elif names[i][1] < j:
            print(title, i, 'duplicated or negative?!')
            i += 1
        else:
            print(title, j, 'missing')
            j += 1


def main():
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'l:t:r:a:n:s:OTH',
                                ['load_path=', 'test_dir=', 'recursions=', 'adversarial=', 'name=',
                                 'steps=', 'overwrite', 'test_only', 'help', 'stats_freq=',
                                 'summary_freq=', 'img_save_freq=', 'test_freq=',
                                 'model_save_freq='])
    except getopt.GetoptError:
        print('Options:')
        print('-n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-n', '--name'):
            c.set_save_name(arg)

    root_dir = os.path.join(c.IMG_SAVE_DIR, 'Tests/')
    for path, dirs, files in os.walk(root_dir):
        files = [file for file in files if os.path.splitext(file)[1] == IMGAE_SUFFIX]
        names = [(os.path.splitext(file)[0], file) for file in files]
        names = [tuple(name.rsplit(sep='_', maxsplit=1)) + (file,) for name, file in names if '_' in name]
        names = [(title, int(nostr), file) for title, nostr, file in names if title and nostr.isdigit()]
        titles = set([name[0] for name in names])
        for title1 in titles:
            names_of_title1 = [(title, no, file) for title, no, file in names if title == title1]
            names_of_title1.sort(key=lambda x: x[1])
            check_numbers(names_of_title1)
            input_filenames = [file for _, _, file in names_of_title1]
            output_filename = title1 + GIF_SUFFIX
            images_to_gif(path, input_filenames, output_filename)

if __name__ == '__main__':
    main()
