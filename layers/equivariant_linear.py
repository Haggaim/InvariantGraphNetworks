import tensorflow as tf
import numpy as np


def equi_2_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''
    basis_dimension = 15
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # initialization values for variables
        coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
        # define variables
        coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

        m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

        ops_out = ops_2_to_2(inputs, m, normalization=normalization)
        ops_out = tf.stack(ops_out, axis=2)

        output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

        # bias
        diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        all_bias = tf.get_variable('all_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0), diag_bias)
        output = output + all_bias + mat_diag_bias

        return output


def equi_2_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m tensor
    '''
    basis_dimension = 5
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # initialization values for variables
        coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
        # define variables
        coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

        m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

        ops_out = ops_2_to_1(inputs, m, normalization=normalization)
        ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

        output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

        # bias
        bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
        output = output + bias

        return output


def equi_1_to_2(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m x m tensor
    '''
    basis_dimension = 5
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # initialization values for variables
        coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
        # define variables
        coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

        m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

        ops_out = ops_1_to_2(inputs, m, normalization=normalization)
        ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m x m

        output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

        # bias
        bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        output = output + bias

        return output


def equi_1_to_1(name, input_depth, output_depth, inputs, normalization='inf', normalization_val=1.0):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m tensor
    '''
    basis_dimension = 2
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # initialization values for variables
        coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
        # define variables
        coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

        m = tf.to_int32(tf.shape(inputs)[2])  # extract dimension

        ops_out = ops_1_to_1(inputs, m, normalization=normalization)
        ops_out = tf.stack(ops_out, axis=2)  # N x D x B x m

        output = tf.einsum('dsb,ndbi->nsi', coeffs, ops_out)  # N x S x m

        # bias
        bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
        output = output + bias

        return output


def equi_basic(name, input_depth, output_depth, inputs):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''
    basis_dimension = 4
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        # initialization values for variables
        coeffs_values = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        #coeffs_values = tf.random_normal([input_depth, output_depth, basis_dimension], dtype=tf.float32)
        # define variables
        coeffs = tf.get_variable('coeffs', initializer=coeffs_values)

        m = tf.to_int32(tf.shape(inputs)[3])  # extract dimension
        float_dim = tf.to_float(m)


        # apply ops
        ops_out = []
        # w1 - identity
        ops_out.append(inputs)
        # w2 - sum cols
        sum_of_cols = tf.divide(tf.reduce_sum(inputs, axis=2), float_dim)  # N x D x m
        ops_out.append(tf.tile(tf.expand_dims(sum_of_cols, axis=2), [1, 1, m, 1]))  # N x D x m x m
        # w3 - sum rows
        sum_of_rows = tf.divide(tf.reduce_sum(inputs, axis=3), float_dim)  # N x D x m
        ops_out.append(tf.tile(tf.expand_dims(sum_of_rows, axis=3), [1, 1, 1, m]))  # N x D x m x m
        # w4 - sum all
        sum_all = tf.divide(tf.reduce_sum(sum_of_rows, axis=2), tf.square(float_dim))  # N x D
        ops_out.append(tf.tile(tf.expand_dims(tf.expand_dims(sum_all, axis=2), axis=3), [1, 1, m, m]))  # N x D x m x m

        ops_out = tf.stack(ops_out, axis=2)
        output = tf.einsum('dsb,ndbij->nsij', coeffs, ops_out)  # N x S x m x m

        # bias
        bias = tf.get_variable('bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        output = output + bias

        return output


def ops_2_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = tf.matrix_diag_part(inputs)   # N x D x m
    sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
    sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
    sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
    sum_all = tf.reduce_sum(sum_of_rows, axis=2)  # N x D

    # op1 - (1234) - extract diag
    op1 = tf.matrix_diag(diag_part)  # N x D x m x m

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = tf.matrix_diag(tf.tile(sum_diag_part, [1, 1, dim]))  # N x D x m x m

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = tf.matrix_diag(sum_of_rows)  # N x D x m x m

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = tf.matrix_diag(sum_of_cols)  # N x D x m x m

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = tf.matrix_diag(tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim]))  # N x D x m x m

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = tf.tile(tf.expand_dims(sum_of_cols, axis=3), [1, 1, 1, dim])  # N x D x m x m

    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = tf.tile(tf.expand_dims(sum_of_rows, axis=3), [1, 1, 1, dim])  # N x D x m x m

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = tf.tile(tf.expand_dims(sum_of_cols, axis=2), [1, 1, dim, 1])  # N x D x m x m

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = tf.tile(tf.expand_dims(sum_of_rows, axis=2), [1, 1, dim, 1])  # N x D x m x m

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs  # N x D x m x m

    # op11 - (1234) + (13)(24) - transpose
    op11 = tf.transpose(inputs, [0, 1, 3, 2])  # N x D x m x m

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = tf.tile(tf.expand_dims(diag_part, axis=3), [1, 1, 1, dim])  # N x D x m x m

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = tf.tile(tf.expand_dims(diag_part, axis=2), [1, 1, dim, 1])  # N x D x m x m

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = tf.tile(tf.expand_dims(sum_diag_part, axis=3), [1, 1, dim, dim])   # N x D x m x m

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = tf.tile(tf.expand_dims(tf.expand_dims(sum_all, axis=2), axis=3), [1, 1, dim, dim])  # N x D x m x m

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)
            op3 = tf.divide(op3, float_dim)
            op4 = tf.divide(op4, float_dim)
            op5 = tf.divide(op5, float_dim**2)
            op6 = tf.divide(op6, float_dim)
            op7 = tf.divide(op7, float_dim)
            op8 = tf.divide(op8, float_dim)
            op9 = tf.divide(op9, float_dim)
            op14 = tf.divide(op14, float_dim)
            op15 = tf.divide(op15, float_dim**2)

    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]


def ops_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = tf.matrix_diag_part(inputs)  # N x D x m
    sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
    sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
    sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=(2, 3))  # N x D

    # op1 - (123) - extract diag
    op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    op2 = tf.tile(sum_diag_part, [1, 1, dim])  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim])  # N x D x m


    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)
            op3 = tf.divide(op3, float_dim)
            op4 = tf.divide(op4, float_dim)
            op5 = tf.divide(op5, float_dim ** 2)

    return [op1, op2, op3, op4, op5]


def ops_1_to_2(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

    # op1 - (123) - place on diag
    op1 = tf.matrix_diag(inputs)  # N x D x m x m

    # op2 - (123) + (12)(3) - tile sum on diag
    op2 = tf.matrix_diag(tf.tile(sum_all, [1, 1, dim]))  # N x D x m x m

    # op3 - (123) + (13)(2) - tile element i in row i
    op3 = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, dim, 1])  # N x D x m x m

    # op4 - (123) + (23)(1) - tile element i in col i
    op4 = tf.tile(tf.expand_dims(inputs, axis=3), [1, 1, 1, dim])  # N x D x m x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = tf.tile(tf.expand_dims(sum_all, axis=3), [1, 1, dim, dim])  # N x D x m x m

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)
            op5 = tf.divide(op5, float_dim)

    return [op1, op2, op3, op4, op5]


def ops_1_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m
    sum_all = tf.reduce_sum(inputs, axis=2, keepdims=True)  # N x D x 1

    # op1 - (12) - identity
    op1 = inputs  # N x D x m

    # op2 - (1)(2) - tile sum of all
    op2 = tf.tile(sum_all, [1, 1, dim])  # N x D x m

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op2 = tf.divide(op2, float_dim)

    return [op1, op2]

