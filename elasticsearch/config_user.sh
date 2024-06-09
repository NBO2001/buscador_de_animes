#!/bin/bash

curl --location -f -X POST "http://localhost:9200/_security/user/kibana_system/_password" -H 'Content-Type: application/json' -H 'Authorization: Basic ZWxhc3RpYzoxMjNjaGFuZ2UuLi4=' --data-raw '{ "password" : "123change..." }'


curl -u elastic:123change... -X POST "localhost:9200/_security/role/app_role" -H 'Content-Type: application/json' -d'
{
  "cluster": [ "all", "manage", "monitor"],
  "indices": [
    {
      "names": [ "anime" ],
      "privileges": ["create", "monitor", "manage", "all", "create_index"]
    }
  ]
}'


curl -u elastic:123change... -X POST "localhost:9200/_security/user/py_elastic" -H 'Content-Type: application/json' -d'
{
  "password" : "py_elastic123",
  "roles" : ["app_role"]
}'