"""Convert image+label dataset into TFRecords (optional).

This script will attempt to import TensorFlow. If TensorFlow is not installed, it will
print instructions for installing it and then exit.

Usage:
    python dataset_export_tfrecords.py --input-dir data/training --output data/train.tfrecord

The script expects files organized by category with paired .png and .txt files, or it
can consume a manifest.csv produced by the generator (image,label,category).
"""
import os
import argparse


def _bytes_feature(value):
    import tensorflow as tf
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_label_pairs_from_dir(root_dir):
    pairs = []
    for cat in os.listdir(root_dir):
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith('.png'):
                img_path = os.path.join(cat_dir, fname)
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        label = f.read().strip()
                    pairs.append((img_path, label))
    return pairs


def image_label_pairs_from_manifest(manifest_csv):
    import csv
    pairs = []
    with open(manifest_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            pairs.append((r['image'], r.get('label', '')))
    return pairs


def convert_to_tfrecord(pairs, out_path):
    try:
        import tensorflow as tf
    except Exception:
        print('TensorFlow not available. Install it with: pip install tensorflow')
        raise

    with tf.io.TFRecordWriter(out_path) as writer:
        for img_path, label in pairs:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            feature = {
                'image_raw': _bytes_feature(img_bytes),
                'label': _bytes_feature(label.encode('utf-8')),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    print(f'Wrote TFRecord to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='Input dataset directory (structured by category)')
    parser.add_argument('--manifest', help='Manifest CSV file produced by generator')
    parser.add_argument('--output', required=True, help='Output TFRecord path')
    args = parser.parse_args()

    if args.manifest:
        pairs = image_label_pairs_from_manifest(args.manifest)
    elif args.input_dir:
        pairs = image_label_pairs_from_dir(args.input_dir)
    else:
        parser.error('Provide --input-dir or --manifest')

    convert_to_tfrecord(pairs, args.output)


if __name__ == '__main__':
    main()
