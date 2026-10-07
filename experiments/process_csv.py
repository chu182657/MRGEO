import csv
import re

def modify_csv(input_filename, output_filename):
    # 建立映射表
    # 规则: 2改成3，3改成2，6改成9，9改成8，0改成1，1改成0
    # 前500条：5改成3
    trans_first_500 = str.maketrans('2369015', '3298103')
    # 500条之后：5改成7
    trans_after_500 = str.maketrans('2369015', '3298107')

    def replace_decimal(match, current_row_count):
        int_part = match.group(1)   # 小数点前的整数部分
        dec_part = match.group(2)   # 小数点后的小数部分
        
        # 根据数据条数（行数）选择对应的映射表修改小数部分
        if current_row_count <= 500:
            new_dec_part = dec_part.translate(trans_first_500)
        else:
            new_dec_part = dec_part.translate(trans_after_500)
            
        return f"{int_part}.{new_dec_part}"

    try:
        # 使用 utf-8-sig 编码读写，可以完美兼容含有中文的 CSV 并防止 Excel 打开时乱码
        with open(input_filename, 'r', encoding='utf-8-sig') as f_in, \
             open(output_filename, 'w', encoding='utf-8-sig', newline='') as f_out:
            
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            
            # 读取并写入表头，表头不算入500条数据内
            header = next(reader, None)
            if header:
                writer.writerow(header)
            
            row_count = 1
            for row in reader:
                new_row = []
                for cell in row:
                    # 使用正则匹配所有带小数点的数字（如 0.0, 3.8571）并执行替换逻辑
                    new_cell = re.sub(r'(\d+)\.(\d+)', lambda m: replace_decimal(m, row_count), cell)
                    new_row.append(new_cell)
                
                writer.writerow(new_row)
                row_count += 1

        print(f"处理完成！共处理了 {row_count - 1} 条数据。")
        print(f"修改后的文件已保存为: {output_filename}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_filename}'，请确保脚本和表格在同一个文件夹内。")

if __name__ == '__main__':
    # 执行修改，读取 111.csv，输出 modified_111.csv
    modify_csv('111.csv', 'modified_111.csv')