# -*- coding: utf-8 -*-

import unittest
import asyncio
from elasticsearch_async import AsyncElasticsearch


class ESAsyncTest(unittest.TestCase):
    # @unittest.skip('Local ES server, can NOT connected!')
    def test_es_info(self):
        async def print_info(client):
            info = await client.info()
            print(info)

        async def transport_close(client):
            await client.transport.close()

        async def main():
            client = AsyncElasticsearch(hosts=['http://192.168.1.148:9210'])
            await print_info(client)
            await transport_close(client)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        # loop.close()

    # @unittest.skip('Local ES server, can NOT connected!')
    def test_es_search(self):
        async def search(client):
            doc = await client.search(index='tracks', doc_type='actsharedetail', body={'size': 2})
            print(doc)

        async def transport_close(transport):
            await transport.close()

        async def main():
            client = AsyncElasticsearch(hosts=['http://192.168.1.148:9210'])
            await search(client)
            await transport_close(client.transport)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        # loop.close()

    def test_es_insert(self):
        async def insert(client):
            rst=await client.insert()
