# -*- coding: utf-8 -*-

import unittest
import asyncio
import pymongo
from bson import ObjectId
import motor.motor_asyncio


class MotorTest(unittest.TestCase):
    mongo_url = 'mongodb://192.168.1.2:27011,192.168.1.2:27012,192.168.1.2:27013/?replicaSet=rs'

    @unittest.skip('Local MongoDB, can NOT connected!')
    def test_client(self):
        client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_url)
        db = client['test-db-dev']
        collection = db['test']

        async def do_find_one():
            c = collection.find_one({'_id': ObjectId('5bf27c99c370b9f37d370000')}, {'_id': 1})
            document = await c
            print(document)

        async def do_find():
            async for document in collection.find({'keyword': 'name'}, {'_id': 1}):
                print(document)

        async def main():
            await do_find_one()
            await do_find()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())

    @unittest.skip('Local MongoDB, can NOT connected!')
    def test_tial(self):
        client = motor.motor_asyncio.AsyncIOMotorClient(self.mongo_url)
        db_test = client['test-db-dev']
        db_local = client['local']

        async def find_by_id(ns, _id):
            coll = db_test[ns.split('.')[1]]
            c = coll.find_one({'_id': _id})
            return await c

        async def tail_example():
            collection = db_local['oplog.rs']
            cursor = collection.find(cursor_type=pymongo.CursorType.TAILABLE,
                                     no_cursor_timeout=True,
                                     oplog_replay=False,
                                     batch_size=100)
            while True:
                if not cursor.alive:
                    await asyncio.sleep(1)
                    cursor = collection.find(cursor_type=pymongo.CursorType.TAILABLE)

                if await cursor.fetch_next:
                    op = cursor.next_object()
                    print('oplog: ', op)
                    doc = await find_by_id(op.get('ns'), op['o']['_id'] or op['o2']['_id'])
                    print('doc', doc)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(tail_example())
