#!/usr/bin/env python

# DO NOT EDIT TOP PORTION OF SCRIPT
"""Download files from simtk.org archive (archive.simtk.org/traj)"""
__author__ = "Randall J. Radmer"
__version__ = "0.1"

import os
# import http.client as httplib
import httplib


copyDestination=os.getcwd()
simtkServer='archive.simtk.org'
rootHttpSelector='traj'

def deepFileWrite(sourceString, filePath):
    """Save the contents of 'souceString' to a file
       indecated by 'filePath'.
       This will make additional directories, if needed."""
    newDir = os.path.dirname(filePath)
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    f=open(filePath, 'w')
    f.write(sourceString)
    f.close()

def makeRequest(request, urlArgs={}):
    """make the URL GET request.  Returns the body in string"""
    h=httplib.HTTPConnection(simtkServer)
    httpSelector="/%s/%s" % (rootHttpSelector, request)
    sep='?'
    for k,v in urlArgs.items():
        httpSelector="%s%s%s=%s" % (httpSelector, sep, k, v)
        sep='&'
    h.request('GET', httpSelector)
    r=h.getresponse()
    data = r.read()
    if r.status>=400:
        raise Exception(httpSelector, r.reason)
    h.close()
    # return r.read()
    return data

def getFileList(project=''):
    """Get a list of all filenames on archive, or if 'project' is
       defined, then get a list of all filenames in the project"""
    urlArgs={}
    if project: urlArgs['project']=project
    return makeRequest('fileList', urlArgs)


def requestFile(filePath, urlArgs):
    """Get one file from the archive"""
    request='project/%s' % filePath.strip('/')
    return makeRequest(request, urlArgs)

if __name__=='__main__':
    # filesString=getFileList('PROJ3036')
    filesString=getFileList()
    fileList=filesString.split()
    urlArgs={}
    for filePath in fileList:
        print("Check %s" % filePath)

        localFilePath=os.path.abspath(os.path.join(copyDestination, filePath))
        if os.path.exists(localFilePath):
            continue

        projectRunClone, filename = os.path.split(filePath)
        projectRun, clone = os.path.split(projectRunClone)
        project, run = os.path.split(projectRun)
        filePathBase, fileExt=os.path.splitext(filePath)

# EDIT BELOW HERE TO CUSTOMIZE WHAT THE SCRIPT DOWNLOADS

        #Example:
        #The lines below will retrieve frame0.tpr AND frame0.xtc from RUN2
        #within PROJ3036 (from CLONE1 and CLONE2)
        if (project!='PROJ3036' or
            run!='RUN2' or
            (clone!='CLONE1' and clone!='CLONE2') or
            (filename!='frame0.tpr' and filename!='frame0.xtc') ):
            continue

        #Example:
        #Uncomment the lines below to download the 10 initial
        # frames from RUN5, CLONE6 ('xtc' files only)
#        import re
#        try:
#            frameNumber = int(re.sub('[^0-9]', '', filename))
#        except: 
#            continue
#        if (project!='PROJ3036' or
#            run!='RUN5' or
#            clone!='CLONE6' or
#            fileExt!='.xtc' or
#            frameNumber>=10 ):
#            continue

        #Example:
        #Uncomment the follwing line to download all files
        #of type 'tpr' and 'xtc'
        #WARNING: very, very Big!
#        if (fileExt!='.tpr' and fileExt!='.xtc'): continue

        s=requestFile(filePath, urlArgs)
        deepFileWrite(s, localFilePath)
        print("Saved %s" % filePath)

        #HOW TO GET DIFFERENT FORMATS DOWNLOADED?

        #If this reqest was for an XTC file, you can request the
        #following options:
        # 'format':'protein-PDB'
        #        Returns the trajectory of the protein in PDB format

        #  Replace 'protein-PDB' (as in example below) by:
        # 'backbone-PDB'
        #        Returns the trajectory of the three protein backbone atoms||

        # Other options:
        # 'mainchain-PDB'
        #        Returns the trajectory of the four protein mainchain atoms||
        # 'protein-H-PDB'
        #        Returns the trajectory of the non-hydrgen protein atoms||
        # 'C-alpha-PDB'
        #        Returns the trajectory of the C-alpha atoms||

        # To get a list of Phi-Psi angles instead of a PDB file,
        # use the following (generated using g_rama):
        # 'format':'protein-PDB'


        #Example:
        #Uncomment the following lines to download a PDB
        #files iwth the backbone atoms
#        if fileExt=='.xtc':
#            s=requestFile(filePath, {'format':'backbone-PDB'})
#            localPdbFilePath='%spdb.gz' % localFilePath[:-3]
#            deepFileWrite(s, localPdbFilePath)
#            print("Saved %s as %s" % (filePath, localPdbFilePath))


