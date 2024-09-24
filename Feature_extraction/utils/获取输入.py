import math

def get_index(img_slide,pred_tile_size,writer_tile_size,area_perc):
    assert pred_tile_size > writer_tile_size ,'要求写入块小于预测块'
    RepetitiveLength = int((1 - math.sqrt(area_perc)) * pred_tile_size / 2)
    W_ys, H_ys = img_slide.get_level_dimension(0)
    dimensions = img_slide.get_level_dimension(0)
    w_count = math.ceil(dimensions[0] / writer_tile_size)
    h_count = math.ceil(dimensions[1] / writer_tile_size)
    W_RowOver = W_ys - (w_count - 1) * writer_tile_size
    ColumnOver = H_ys - (h_count - 1) * writer_tile_size
    input_img_index_list = []
    title_index = {}
    w_h_list=[]

    index_uuuu = 0

    for h_tmp in range(h_count):
        for w_tmp in range(w_count):
            if (h_tmp == 0):  # 处理第0行数据
                if (w_tmp == 0):  # 针对第一个块进行特殊处理
                    input_img_index_list.append([0, 0,index_uuuu])
                    title_index[index_uuuu]=[True, 0, writer_tile_size, 0, writer_tile_size]
                elif (w_tmp == w_count - 1):  # 对最后一个块进行特征处理
                    input_img_index_list.append([W_ys - pred_tile_size, 0,index_uuuu])
                    title_index[index_uuuu]=[False, 0, writer_tile_size, 0, W_RowOver, 0, writer_tile_size, pred_tile_size - W_RowOver,pred_tile_size]
                else:
                    # 对中间块进行处理
                    input_img_index_list.append([w_tmp * writer_tile_size - RepetitiveLength, 0,index_uuuu])
                    title_index[index_uuuu]=[True, 0, writer_tile_size, RepetitiveLength, writer_tile_size + RepetitiveLength]
            elif (h_tmp == h_count - 1):  # 处理最后一行数据
                if (w_tmp == 0):  # 对第一个块进行特殊处理
                    input_img_index_list.append([0, H_ys - pred_tile_size,index_uuuu])
                    title_index[index_uuuu]=[False, 0, ColumnOver, 0, writer_tile_size, pred_tile_size - ColumnOver, pred_tile_size, 0,writer_tile_size]
                elif (w_tmp == w_count - 1):  # 对最后一个块进行特征处理
                    input_img_index_list.append([W_ys - pred_tile_size, H_ys - pred_tile_size,index_uuuu])
                    title_index[index_uuuu]=[False, 0, ColumnOver, 0, W_RowOver, pred_tile_size - ColumnOver, pred_tile_size,pred_tile_size - W_RowOver, pred_tile_size]
                else:  # 对中间块进行处理
                    input_img_index_list.append([w_tmp * writer_tile_size - RepetitiveLength, H_ys - pred_tile_size,index_uuuu])
                    title_index[index_uuuu]=[False, 0, ColumnOver, 0, writer_tile_size, pred_tile_size - ColumnOver, pred_tile_size, RepetitiveLength, writer_tile_size + RepetitiveLength]
            else:  # 处理第1-n行数据
                if (w_tmp == 0):  ##对第一个块进行特殊处理
                    input_img_index_list.append([0, h_tmp * writer_tile_size - RepetitiveLength,index_uuuu])
                    title_index[index_uuuu]=[True, RepetitiveLength, writer_tile_size + RepetitiveLength, 0, writer_tile_size]
                elif (w_tmp == w_count - 1):  # 对最后一个块进行特征处理
                    input_img_index_list.append([W_ys - pred_tile_size, h_tmp * writer_tile_size - RepetitiveLength,index_uuuu])
                    title_index[index_uuuu]=[False, 0, writer_tile_size, 0, W_RowOver, RepetitiveLength,writer_tile_size + RepetitiveLength, pred_tile_size - W_RowOver,pred_tile_size]
                else:  # 对中间块进行处理
                    input_img_index_list.append([w_tmp * writer_tile_size - RepetitiveLength, h_tmp * writer_tile_size - RepetitiveLength,index_uuuu])
                    title_index[index_uuuu]=[True, RepetitiveLength, writer_tile_size + RepetitiveLength, RepetitiveLength, writer_tile_size + RepetitiveLength]
            w_h_list.append([h_tmp, w_tmp])
            index_uuuu = index_uuuu+ 1

    return input_img_index_list,title_index,w_h_list