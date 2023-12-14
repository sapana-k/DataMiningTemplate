from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def hello(request):
    return Response({'msg': 'Hello, world!'})

@api_view(['POST'])
def hello(request):
    return Response({'msg': 'File at backend'})
    #write code for five number summary

@api_view(['POST'])
def kmeans(request):
    print(request.data['dataset'])
    return Response({'mean' : request.data['dataset']})
    
        