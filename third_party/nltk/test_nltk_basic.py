import unittest


class NLTKBasicTest(unittest.TestCase):
    def test_first_app(self):
        import urllib3
        import nltk
        from bs4 import BeautifulSoup
        url = 'http://python.org/'
        with urllib3.PoolManager() as http:
            response = http.request('GET', url)
        html = response.data
        print(len(html))

        soup = BeautifulSoup(html, 'html')
        clean = soup.get_text()
        tokens = [tok for tok in clean.split()]

        stopwords = [word.strip().lower() for word in open("english.stop.txt")]
        clean_tokens = [tok for tok in tokens if len(tok.lower()) > 1 and (tok.lower() not in stopwords)]
        freq_dist_nltk = nltk.FreqDist(clean_tokens)
        print(freq_dist_nltk)
        for k, v in freq_dist_nltk.items():
            print('%s:%d' % (k, v))
        freq_dist_nltk.plot(50)
