# simple in-memory shared-artifact store (one shared file for all sessions)
_shared_artifact_lock = threading.Lock()
_shared_artifact = {
    "filename": None,
    "mime_type": None,
    "data": None,         # raw bytes
    "uploaded_at": None,  # unix timestamp
}



@app.get("/shared-artifact/upload-form", response_class=HTMLResponse)
async def upload_form():
    return HTMLResponse(
        """
        <html><body>
          <h3>Upload a shared artifact (replaces previous)</h3>
          <form action="/shared-artifact/upload" enctype="multipart/form-data" method="post">
            <input name="file" type="file"/>
            <input type="submit"/>
          </form>
        </body></html>
        """
    )


@app.post("/shared-artifact/upload")
async def upload_shared_artifact(file: UploadFile = File(...)):
    """
    Upload (replace) the single shared artifact.
    Returns metadata (filename, size, mime, timestamp).
    """
    content = await file.read()
    with _shared_artifact_lock:
        _shared_artifact["filename"] = file.filename
        _shared_artifact["mime_type"] = file.content_type or "application/octet-stream"
        _shared_artifact["data"] = content
        _shared_artifact["uploaded_at"] = int(time.time())

    return JSONResponse({
        "status": "ok",
        "filename": file.filename,
        "mime_type": _shared_artifact["mime_type"],
        "size": len(content),
        "uploaded_at": _shared_artifact["uploaded_at"],
    })


@app.get("/shared-artifact")
async def get_shared_artifact_metadata():
    """
    Returns JSON metadata + base64-encoded data: { filename, mime_type, uploaded_at, data_b64 }.
    (This is friendly for agents that expect JSON.)
    """
    with _shared_artifact_lock:
        if _shared_artifact["data"] is None:
            raise HTTPException(status_code=404, detail="No shared artifact uploaded")
        return {
            "filename": _shared_artifact["filename"],
            "mime_type": _shared_artifact["mime_type"],
            "uploaded_at": _shared_artifact["uploaded_at"],
            "data_b64": base64.b64encode(_shared_artifact["data"]).decode("ascii"),
        }


@app.get("/shared-artifact/download")
async def download_shared_artifact():
    """
    Download the current shared artifact as a raw file.
    """
    with _shared_artifact_lock:
        if _shared_artifact["data"] is None:
            raise HTTPException(status_code=404, detail="No shared artifact uploaded")
        bio = BytesIO(_shared_artifact["data"])
        headers = {"Content-Disposition": f'attachment; filename="{_shared_artifact["filename"]}"'}
        return StreamingResponse(bio, media_type=_shared_artifact["mime_type"], headers=headers)


# @app.delete("/shared-artifact")
# async def delete_shared_artifact():
#     """
#     Remove the shared artifact.
#     """
#     with _shared_artifact_lock:
#         if _shared_artifact["data"] is None:
#             return {"status": "no-op", "detail": "no file present"}
#         _shared_artifact.update({"filename": None, "mime_type": None, "data": None, "uploaded_at": None})
#     return {"status": "deleted"}
