import importlib.util, io, tempfile, unittest

class SecurityDependencyCompatibility(unittest.TestCase):
    def test_web_and_token_interfaces(self):
        if not importlib.util.find_spec('flask') and not importlib.util.find_spec('jwt'): self.skipTest('Web/token libraries are not used')
        if importlib.util.find_spec('flask'):
            from flask import Flask, jsonify
            app = Flask(__name__)
            app.add_url_rule('/health', view_func=lambda: jsonify(ok=True))
            self.assertEqual(app.test_client().get('/health').get_json(), {'ok': True})
        if importlib.util.find_spec('jwt'):
            import jwt
            secret = 'local-test-secret-only-at-least-32-bytes'
            token = jwt.encode({'sub': 'fixture'}, secret, algorithm='HS256')
            self.assertEqual(jwt.decode(token, secret, algorithms=['HS256'])['sub'], 'fixture')
    def test_image_round_trip(self):
        if not importlib.util.find_spec('PIL'): self.skipTest('Pillow is not used')
        from PIL import Image
        output = io.BytesIO()
        Image.new('RGB', (2, 2), (12, 34, 56)).save(output, format='PNG')
        output.seek(0)
        self.assertEqual(Image.open(output).getpixel((0, 0)), (12, 34, 56))
    def test_tensor_and_safe_checkpoint_round_trip(self):
        if not importlib.util.find_spec('torch'): self.skipTest('Torch is not used')
        import torch
        torch.set_num_threads(1)
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        self.assertEqual((tensor @ torch.eye(2)).tolist(), tensor.tolist())
        output = io.BytesIO(); torch.save({'weights': tensor}, output); output.seek(0)
        self.assertTrue(torch.equal(torch.load(output, weights_only=True)['weights'], tensor))
        if importlib.util.find_spec('torchvision'):
            from torchvision.ops import box_iou
            box = torch.tensor([[0., 0., 2., 2.]])
            self.assertEqual(box_iou(box, box).item(), 1.)
    def test_small_transformer_forward_pass(self):
        if not importlib.util.find_spec('transformers'): self.skipTest('Transformers is not used')
        import torch
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(vocab_size=8, hidden_size=8, intermediate_size=16, num_attention_heads=2, num_hidden_layers=1))
        output = model(torch.tensor([[1, 2]]))
        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 2, 8))
    def test_keras_backend_forward_pass(self):
        if not importlib.util.find_spec('tensorflow') or not importlib.util.find_spec('keras'): self.skipTest('TensorFlow/Keras are not used')
        import keras, numpy as np
        model = keras.Sequential([keras.layers.Input(shape=(2,)), keras.layers.Dense(1)])
        self.assertEqual(tuple(model(np.ones((1, 2), dtype='float32')).shape), (1, 1))
    def test_http_request_preparation(self):
        if not importlib.util.find_spec('requests'): self.skipTest('Requests is not used')
        import requests
        request = requests.Request('GET', 'https://example.org/path', params={'q': 'a b'}).prepare()
        self.assertEqual(request.url, 'https://example.org/path?q=a+b')

if __name__ == '__main__': unittest.main()
