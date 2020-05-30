from transformerocr.loader.DataLoader import DataGen
from transformerocr.model.vocab import Vocab

def test_loader():
    chars = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

    vocab = Vocab(chars)
    s_gen = DataGen('./transformerocr/tests/', 'sample.txt', vocab, 'cpu')
    iterator = s_gen.gen(30)
    for batch in iterator:
        assert batch['img'].shape[1]==3, 'image must have 3 channels'
        assert batch['img'].shape[2]==32, 'the height must be 32'
        print(batch['img'].shape, batch['tgt_input'].shape, batch['tgt_output'].shape, batch['tgt_padding_mask'].shape)

if __name__ == '__main__':
    test_loader()
