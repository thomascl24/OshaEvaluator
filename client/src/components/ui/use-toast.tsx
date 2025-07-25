import { toast as sonnerToast } from "sonner";
import { X } from "lucide-react"; // Optional icon for the dismiss button

export function useToast() {
  return {
    toast: (props: {
      title: string;
      description?: string;
      variant?: "success" | "error" | "info" | "warning";
    }) => {
      const id = sonnerToast.custom((t) => (
        <div className="w-full max-w-xl bg-white shadow-lg rounded-lg px-8 py-6 flex flex-col gap-3">
          <div className="text-lg font-semibold text-green-700">{props.title}</div>
          {props.description && (
            <div className="text-base text-slate-700">{props.description}</div>
          )}
          <div className="flex justify-end">
            <button
              onClick={() => sonnerToast.dismiss(t)}
              className="mt-2 inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-green-600 hover:bg-green-700 rounded-md"
            >
              <X className="w-4 h-4 mr-2" />
              Dismiss
            </button>
          </div>
        </div>
      ), {
        position: "bottom-center",
        duration: Infinity, // must be dismissed manually
      });
    },
  };
}
