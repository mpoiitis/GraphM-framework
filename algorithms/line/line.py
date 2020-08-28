from .utils import batchgen_train, LINE_loss, transform_data, write_embedding
from .model import create_model


def line(args, G):
    adj_list = transform_data(G)
    epoch_train_size = (((int(len(adj_list)/args.batch_size))*(1 + args.negative_ratio)*args.batch_size) + (1 + args.negative_ratio)*(len(adj_list)%args.batch_size))
    numNodes = len(G.nodes())

    data_gen = batchgen_train(adj_list, numNodes, args.batch_size, args.negative_ratio, args.negative_sampling)

    model, embed_generator = create_model(numNodes, args.dimension)
    model.summary()

    model.compile(optimizer='rmsprop', loss={'dot': LINE_loss})
    model.fit(data_gen, steps_per_epoch=epoch_train_size, epochs=args.iter, verbose=1)

    keys = list(G.nodes())
    write_embedding(args, keys, embed_generator)
