// app/upload/page.tsx
"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export default function UploadPage() {
    const [videoUrl, setVideoUrl] = useState<string | null>(null)
    const [info, setInfo] = useState<any>(null)
    const [file, setFile] = useState<File | null>(null)
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {

        const file = e.target.files?.[0]
        if (file) {
            setFile(file)
            const objectUrl = URL.createObjectURL(file)
            setVideoUrl(objectUrl)
        }
    }
    const handleMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
        const video = e.currentTarget
        setInfo({
            duration: video.duration,
            resolution: `${video.videoWidth}x${video.videoHeight}`,
            name: (file && file.name) || '?'
        })
    }
    const handleUpload = async () => {
        if (!file) return

        const formData = new FormData()
        formData.append("video", file)

        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        })

        const data = await res.json()
        alert("Uploaded: " + data.url)
    }

    return (
        <div className="max-w-md mx-auto mt-10 p-4 border rounded-lg space-y-4">
            <Input
                type="file"
                accept="video/*"
                onChange={(e) => handleFileChange(e)}
            />
            <Button onClick={handleUpload} disabled={!file}>
                Upload Video
            </Button>
            {info && (
                <div className="mt-4 text-sm text-gray-700 dark:text-gray-300">
                    <p>🕒 Duration: {info.duration.toFixed(2)} seconds</p>
                    <p>📐 Resolution: {info.resolution}</p>
                    <p> Name: {info.name}</p>

                </div>
            )}            {videoUrl && (
                <video controls width={720} src={videoUrl}   onLoadedMetadata={handleMetadata} />
            )}
        </div>
    )
}
