import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import multer from "multer";

// Configure multer for file uploads
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Health check endpoint - proxy to FastAPI
  app.get("/app/health", async (req, res) => {
    try {
      const response = await fetch("http://localhost:8000/app/health");
      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      console.error("FastAPI health check failed:", error);
      res.status(503).json({ 
        error: "FastAPI service unavailable",
        details: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Prediction endpoint - proxy to FastAPI
  app.post("/app/predict", upload.single('file'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }

      // Check if file has .json extension
      const allowedExtenstions = ['.json', '.pdf'];
      const hasAllowedExtension = allowedExtenstions.some(ext => 
        req.file.originalname?.toLowerCase().endsWith(ext)
      );

      if (!hasAllowedExtension) {
        return res.status(415).json({ 
          error: "Please upload a .json or .pdf file" 
        });
      }

      // Create FormData to forward to FastAPI
      const formData = new FormData();
      const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
      formData.append('file', blob, req.file.originalname);

      // Forward request to FastAPI service
      const response = await fetch("http://localhost:8000/app/predict", {
        method: 'POST',
        body: formData
      });

      const contentType = response.headers.get("content-type") || "";

      let data;
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        throw new Error(`Non-JSON response from FastAPI: ${text}`);
      }
      
      res.status(response.status).json(data);

    } catch (error) {
      console.error("FastAPI prediction failed:", error);
      res.status(503).json({ 
        error: "FastAPI service unavailable",
        details: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
