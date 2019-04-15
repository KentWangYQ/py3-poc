# -*- coding: utf-8 -*-

import unittest
import asyncio
from elasticsearch_async import AsyncElasticsearch


class ESAsyncTest(unittest.TestCase):
    @unittest.skip('Local ES server, can NOT connected!')
    def test_es_info(self):
        client = AsyncElasticsearch(hosts=['http://192.168.1.148:9210'])

        async def print_info():
            info = await client.info()
            print(info)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(print_info())
        loop.run_until_complete(client.transport.close())
        # loop.close()

    @unittest.skip('Local ES server, can NOT connected!')
    def test_es_search(self):
        client = AsyncElasticsearch(hosts=['http://192.168.1.148:9210'])

        async def search():
            doc = await client.search(index='tracks', body={'size': 2})
            print(doc)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(search())
        loop.run_until_complete(client.transport.close())
        # loop.close()
