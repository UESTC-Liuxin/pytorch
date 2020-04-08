import logging

class Mylog(object):
    """
    @create log file and output log information
    """
    def __init__(self,logFilename):
        logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=logFilename,
                        filemode='a')
        console = logging.StreamHandler()  # 定义console handler
        console.setLevel(logging.INFO)  # 定义该handler级别
        formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')  # 定义该handler格式
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)  # 实例化添加handler


    def debug_out(self,str):
        """
        output debug information
        :param str: information
        :return:
        """
        logging.debug(str)

    def info_out(self,str):
        """
        output running info
        :param str:
        :return:
        """
        logging.info(str)