import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { FileUpload } from "@/components/ui/file-upload";
import { ExpandableText } from "@/components/ui/expandable-text";
import { getEvaluationBadgeClass } from "@/lib/utils";

interface EvaluationResult {
  predictions: string[];
  premises: string[];
  hypotheses: string[];
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [results, setResults] = useState<EvaluationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const evaluationMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/app/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText}${errorText ? ` - ${errorText}` : ''}`);
      }
      
      return response.json();
    },
    onSuccess: (data: EvaluationResult) => {
      setResults(data);
      setError(null);
    },
    onError: (error: Error) => {
      setError(`Failed to process file: ${error.message}`);
      setResults(null);
    },
  });

  const handleRunEvaluation = () => {
    if (selectedFile) {
      evaluationMutation.mutate(selectedFile);
    }
  };

  const handleClearUpload = () => {
    setSelectedFile(null);
    setResults(null);
    setError(null);
    evaluationMutation.reset();
  };

  const dismissError = () => {
    setError(null);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-slate-900 mb-3">Safety Regulation Evaluator</h2>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          Upload your instruction manual to automatically identify relevant OSHA regulations and evaluate compliance requirements.
        </p>
      </div>

      {/* Upload Section */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-xl font-medium">Upload Manual</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <FileUpload
            onFileSelect={setSelectedFile}
            selectedFile={selectedFile}
            disabled={evaluationMutation.isPending}
          />
          
          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              onClick={handleRunEvaluation}
              disabled={!selectedFile || evaluationMutation.isPending}
              className="flex-1 py-3 font-medium"
            >
              {evaluationMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                'Run Evaluation'
              )}
            </Button>
            <Button
              variant="outline"
              onClick={handleClearUpload}
              className="sm:flex-shrink-0 py-3 font-medium"
            >
              Clear & Upload New
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive" className="animate-slide-up">
          <AlertDescription className="flex items-start justify-between">
            <span>{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={dismissError}
              className="text-destructive hover:text-destructive/80 p-0 h-auto ml-3"
            >
              Dismiss
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Results Section */}
      {results && (
        <Card className="shadow-sm animate-fade-in">
          <CardHeader className="border-b">
            <div className="flex items-center justify-between">
              <CardTitle className="text-xl font-medium">Evaluation Results</CardTitle>
              <span className="text-sm text-slate-500">
                {results.predictions.length} evaluations
              </span>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Instruction Step
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      OSHA Regulation
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Evaluation
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-200">
                  {results.predictions.map((prediction, index) => {
                    const premise = results.premises[index] || '';
                    const hypothesis = results.hypotheses[index] || '';
                    
                    return (
                      <tr key={index} className="hover:bg-slate-50">
                        <td className="px-6 py-4 text-sm text-slate-900">
                          <ExpandableText text={premise} />
                        </td>
                        <td className="px-6 py-4 text-sm text-slate-900">
                          <ExpandableText text={hypothesis} />
                        </td>
                        <td className="px-6 py-4">
                          <Badge className={getEvaluationBadgeClass(prediction)}>
                            {prediction}
                          </Badge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
