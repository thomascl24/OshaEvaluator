import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import examplePDF from "../docs/example.pdf";

export default function About() {
  const example = {
    Title: "Replacing a Car Tire",
    Steps: ["Step 1...", "Step 2..."]
  }
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-slate-900 mb-3">About</h2>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          Learn more about the OSHA Compliance Evaluator and how it works.
        </p>
      </div>

      {/* Back to Home Button */}
      <div>
        <Link href="/">
          <Button variant="outline" className="mb-6">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </Link>
      </div>

      {/* Content */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl font-light">OSHA Compliance Evaluator</CardTitle>
        </CardHeader>
        <CardContent className="prose prose-slate max-w-none space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-slate-900 mb-4">Model Overview</h3>
            <p className="text-slate-700 leading-relaxed">
              The OSHA Compliance Evaluator uses advanced Natural Language Inference (NLI) models to automatically 
              identify relevant safety regulations for instruction manual steps. The system leverages RoBERTa-based 
              models trained on safety compliance data to provide accurate regulatory guidance.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-slate-900 mb-3">How It Works</h4>
            <ol className="list-decimal list-inside text-slate-700 space-y-2 leading-relaxed">
              <li>Upload your instruction manual in JSONL or PDF format</li>
                  <ul className="list-disc pl-6 mt-1 space-y-1 space-x-1">
                    <li>JSONL files are flexible and allow the user to input multiple manuals with each new line having a dictionary with the following format:</li>
                    <pre className="bg-slate-700 text-white text-sm p-4 rounded-md overflow-x-auto whitespace-pre">
                      <code>{JSON.stringify(example, null, 4)}</code>
                    </pre>
                    <li>PDF manuals will be processed in a line by line format, where each line will be considered a step (pictures will not be processed)
                    an example PDF manual is shown below:  
                    </li>
                    <embed
                      src={examplePDF}           /* or a remote URL / Blob URL */
                      type="application/pdf"
                      width="100%"                       /* full‑width */
                      height="500"                       /* fixed height – make this responsive if you like */
                      className="rounded-md shadow"
                    />
                  </ul>
              <li>The system extracts and processes individual instruction steps</li>
              <li>Each step is analyzed against a comprehensive OSHA regulation database</li>
              <li>Relevant regulations are identified and compliance evaluations are generated</li>
              <li>Results are presented in an easy-to-review format</li>
            </ol>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-slate-900 mb-3">Training Data</h4>
            <p className="text-slate-700 leading-relaxed">
              The model has been trained on a curated dataset of instruction manuals paired with corresponding 
              OSHA regulations. The training process includes natural language inference tasks to ensure accurate 
              regulatory mapping and compliance assessment.
            </p>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-slate-900 mb-3">Evaluation Types</h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-slate-700"><strong>Entailment:</strong> The instruction step directly requires compliance with the regulation</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-slate-700"><strong>Contradiction:</strong> The instruction step conflicts with the regulation</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="text-slate-700"><strong>Neutral:</strong> The instruction step is neither required nor prohibited by the regulation</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-slate-900 mb-3">Version History</h4>
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.1.1</p>
                <p className="text-sm text-slate-600">Added new Team page and further enhanced About page</p>
                <p className="text-xs text-slate-500">Released: July 2025</p>
              </div>
            </div> 
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.1.0</p>
                <p className="text-sm text-slate-600">Updated Predictor class with OpenAI API</p>
                <p className="text-xs text-slate-500">Released: July 2025</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.0.3</p>
                <p className="text-sm text-slate-600">Enhanced frontend with About page and tabular output on home page</p>
                <p className="text-xs text-slate-500">Released: July 2025</p>
              </div>
            </div> 
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.0.2</p>
                <p className="text-sm text-slate-600">Improved RAG with threshold improvements</p>
                <p className="text-xs text-slate-500">Released: July 2025</p>
              </div>
            </div>
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.0.1</p>
                <p className="text-sm text-slate-600">Refactored pipeline and bug fixes</p>
                <p className="text-xs text-slate-500">Released: July 2025</p>
              </div>
            </div>              
            <div className="space-y-3">
              <div className="border-l-4 border-primary pl-4">
                <p className="font-medium text-slate-900">Version 1.0.0</p>
                <p className="text-sm text-slate-600">Initial release with RoBERTa-large model and vector store integration</p>
                <p className="text-xs text-slate-500">Released: June 2025</p>
              </div>
            </div>
          </div>

          <div className="mt-8 p-4 bg-slate-50 rounded-lg border-l-4 border-yellow-400">
            <p className="text-sm text-slate-700">
              <strong>Important Notice:</strong> This tool provides guidance for OSHA compliance but should not replace 
              professional safety consultation. Always verify regulatory requirements with official OSHA documentation 
              and consult qualified safety professionals for critical compliance decisions.
            </p>
          </div>

          <div>
            <h4 className="text-lg font-semibold text-slate-900 mb-3">Technical Specifications</h4>
            <ul className="list-disc list-inside text-slate-700 space-y-1 leading-relaxed">
              <li>Base Model: RoBERTa-Large (355M parameters)</li>
              <li>Vector Store: Qdrant for regulation embeddings</li>
              <li>Embedding Model: HuggingFace Sentence Transformers</li>
              <li>Training Framework: PyTorch Lightning</li>
              <li>Inference Engine: FastAPI with Redis caching</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
